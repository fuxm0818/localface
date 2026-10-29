use std::fs::{self, File};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[cfg(unix)]
use std::sync::atomic::{AtomicBool, Ordering};

use tauri::menu::{CheckMenuItem, Menu, MenuItem};
use tauri::tray::TrayIconBuilder;
use tauri::{AppHandle, Manager, RunEvent};

#[cfg(target_os = "macos")]
use objc2_app_kit::{NSColor, NSWindow, NSWindowCollectionBehavior, NSScreenSaverWindowLevel};

#[cfg(target_os = "macos")]
const CAMERA_SETTINGS_URL: &str =
    "x-apple.systempreferences:com.apple.preference.security?Privacy_Camera";

#[cfg(target_os = "macos")]
fn macos_camera_status() -> &'static str {
    use objc2_av_foundation::{AVAuthorizationStatus, AVCaptureDevice, AVMediaTypeVideo};

    let Some(media) = (unsafe { AVMediaTypeVideo }) else {
        return "unknown";
    };
    let status = unsafe { AVCaptureDevice::authorizationStatusForMediaType(media) };
    if status == AVAuthorizationStatus::NotDetermined {
        "not_determined"
    } else if status == AVAuthorizationStatus::Restricted {
        "restricted"
    } else if status == AVAuthorizationStatus::Denied {
        "denied"
    } else if status == AVAuthorizationStatus::Authorized {
        "authorized"
    } else {
        "unknown"
    }
}

#[tauri::command]
fn camera_permission_status() -> String {
    #[cfg(target_os = "macos")]
    {
        macos_camera_status().to_string()
    }
    #[cfg(not(target_os = "macos"))]
    {
        "unknown".to_string()
    }
}

#[tauri::command]
async fn request_camera_permission() -> bool {
    #[cfg(target_os = "macos")]
    {
        use block2::RcBlock;
        use objc2::runtime::Bool;
        use objc2_av_foundation::{AVCaptureDevice, AVMediaTypeVideo};

        let Some(media) = (unsafe { AVMediaTypeVideo }) else {
            return false;
        };
        let rx = {
            let (tx, rx) = tokio::sync::oneshot::channel();
            let tx = Mutex::new(Some(tx));
            let handler = RcBlock::new(move |granted: Bool| {
                if let Some(tx) = tx.lock().unwrap().take() {
                    let _ = tx.send(granted.as_bool());
                }
            });
            unsafe {
                AVCaptureDevice::requestAccessForMediaType_completionHandler(media, &handler);
            }
            // AVFoundation 会复制 completion block，本地这份可以释放
            drop(handler);
            rx
        };
        rx.await.unwrap_or(false)
    }
    #[cfg(not(target_os = "macos"))]
    {
        false
    }
}

#[tauri::command]
fn open_camera_settings() {
    #[cfg(target_os = "macos")]
    {
        let _ = std::process::Command::new("open").arg(CAMERA_SETTINGS_URL).spawn();
    }
}

#[cfg(unix)]
static SHUTDOWN: AtomicBool = AtomicBool::new(false);

#[cfg(unix)]
extern "C" fn handle_signal(_signal: libc::c_int) {
    SHUTDOWN.store(true, Ordering::SeqCst);
}

#[cfg(unix)]
fn install_signal_handlers() {
    unsafe {
        libc::signal(libc::SIGTERM, handle_signal as extern "C" fn(libc::c_int) as libc::sighandler_t);
        libc::signal(libc::SIGINT, handle_signal as extern "C" fn(libc::c_int) as libc::sighandler_t);
    }
}

#[derive(Clone)]
struct Backend {
    child: Arc<Mutex<Option<Child>>>,
    port: u16,
    token: String,
    log_path: PathBuf,
    pid_path: PathBuf,
}

fn free_port() -> u16 {
    if let Ok(raw) = std::env::var("FACE_PORT") {
        if let Ok(port) = raw.parse::<u16>() {
            return port;
        }
    }
    TcpListener::bind(("127.0.0.1", 0))
        .and_then(|l| l.local_addr())
        .map(|addr| addr.port())
        .unwrap_or(5273)
}

fn make_token() -> String {
    let mut seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x9e37_79b9)
        ^ ((std::process::id() as u64) << 32);
    let mut token = String::with_capacity(64);
    for _ in 0..4 {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        token.push_str(&format!("{seed:016x}"));
    }
    token
}

fn loopback(port: u16) -> SocketAddr {
    SocketAddr::from(([127, 0, 0, 1], port))
}

fn health_ok(port: u16) -> bool {
    let Ok(mut stream) = TcpStream::connect_timeout(&loopback(port), Duration::from_millis(400)) else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_millis(1200)));
    let _ = stream.set_write_timeout(Some(Duration::from_millis(1200)));
    let request = b"GET /api/state HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n";
    if stream.write_all(request).is_err() {
        return false;
    }
    let mut buf = [0u8; 48];
    match stream.read(&mut buf) {
        Ok(n) if n > 0 => String::from_utf8_lossy(&buf[..n]).starts_with("HTTP/1.1 200"),
        _ => false,
    }
}

fn request_shutdown(port: u16, token: &str) {
    let Ok(mut stream) = TcpStream::connect_timeout(&loopback(port), Duration::from_millis(300)) else {
        return;
    };
    let _ = stream.set_write_timeout(Some(Duration::from_millis(500)));
    let _ = stream.set_read_timeout(Some(Duration::from_millis(500)));
    let request = format!(
        "POST /api/shutdown HTTP/1.1\r\nHost: 127.0.0.1\r\nX-Face-Shutdown-Token: {token}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
    );
    let _ = stream.write_all(request.as_bytes());
    let mut scratch = [0u8; 64];
    let _ = stream.read(&mut scratch);
}

fn http_request(port: u16, method: &str, path: &str, body: Option<&str>) -> Option<String> {
    let mut stream = TcpStream::connect_timeout(&loopback(port), Duration::from_millis(1500)).ok()?;
    let _ = stream.set_write_timeout(Some(Duration::from_millis(1500)));
    let _ = stream.set_read_timeout(Some(Duration::from_millis(3000)));
    let body = body.unwrap_or("");
    let request = format!(
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(request.as_bytes()).ok()?;
    let mut buf = String::new();
    stream.read_to_string(&mut buf).ok()?;
    buf.split_once("\r\n\r\n").map(|(_, body)| body.to_string())
}

fn guard_enabled_of(port: u16) -> Option<bool> {
    // 服务端可能正忙着做人脸推理，读两次提高成功率
    for _ in 0..2 {
        if let Some(body) = http_request(port, "GET", "/api/guard", None) {
            if body.contains("\"enabled\":true") {
                return Some(true);
            }
            if body.contains("\"enabled\":false") {
                return Some(false);
            }
        }
        std::thread::sleep(Duration::from_millis(250));
    }
    None
}

fn project_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."))
}

#[cfg(unix)]
fn enter_process_group(command: &mut Command) {
    use std::os::unix::process::CommandExt;
    command.process_group(0);
}

#[cfg(not(unix))]
fn enter_process_group(_command: &mut Command) {}

#[cfg(not(debug_assertions))]
fn sidecar_path(app: &AppHandle) -> Option<PathBuf> {
    let exe_dir = std::env::current_exe().ok()?.parent()?.to_path_buf();
    let resource_dir = app.path().resource_dir().ok();
    let name = if cfg!(windows) { "localface-server.exe" } else { "localface-server" };
    let mut candidates = vec![exe_dir.join(name)];
    if let Some(dir) = resource_dir {
        candidates.push(dir.join("bin").join(name));
        candidates.push(dir.join(name));
    }
    candidates.into_iter().find(|p| p.exists())
}

struct Launch {
    program: PathBuf,
    args: Vec<PathBuf>,
    cwd: PathBuf,
    static_dir: PathBuf,
    models_dir: PathBuf,
}

fn resolve_launch(app: &AppHandle) -> Result<Launch, String> {
    let root = project_root();
    #[cfg(debug_assertions)]
    {
        let _ = app;
        let models = root.join("models");
        if !models.join("arcface_r50.onnx").exists() {
            return Err(format!("开发模式下找不到模型目录：{}", models.display()));
        }
        return Ok(Launch {
            program: std::env::var("FACE_NODE").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from("node")),
            args: vec![root.join("server.mjs")],
            cwd: root.clone(),
            static_dir: root,
            models_dir: models,
        });
    }

    #[cfg(not(debug_assertions))]
    {
        let _ = root;
        let resource_dir = app
            .path()
            .resource_dir()
            .map_err(|e| format!("找不到应用资源目录：{e}"))?;
        let program = sidecar_path(app).ok_or_else(|| "安装包缺少 Node 运行时（localface-server）".to_string())?;
        let script = resource_dir.join("server").join("server.mjs");
        if !script.exists() {
            return Err(format!("安装包缺少服务代码：{}", script.display()));
        }
        Ok(Launch {
            program,
            args: vec![script],
            cwd: resource_dir.join("server"),
            static_dir: resource_dir.join("static"),
            models_dir: resource_dir.join("models"),
        })
    }
}

fn cleanup_previous(data_dir: &Path) {
    let path = data_dir.join("logs").join("backend.pid");
    if let Ok(raw) = fs::read_to_string(&path) {
        if let Ok(pid) = raw.trim().parse::<i32>() {
            if pid > 1 && pid != std::process::id() as i32 {
                #[cfg(unix)]
                unsafe {
                    if libc::kill(pid, 0) == 0 {
                        libc::kill(-pid, libc::SIGTERM);
                        std::thread::sleep(Duration::from_millis(600));
                        libc::kill(-pid, libc::SIGKILL);
                    }
                }
            }
        }
    }
    let _ = fs::remove_file(&path);
}

fn start_backend(app: &AppHandle, launch: Launch, port: u16, token: &str) -> Result<(Child, PathBuf, PathBuf), String> {
    // 允许用 FACE_DATA_DIR 指定数据目录：便于验证「全新首启」或做便携模式
    let data_dir = match std::env::var("FACE_DATA_DIR") {
        Ok(dir) if !dir.trim().is_empty() => PathBuf::from(dir),
        _ => app
            .path()
            .app_data_dir()
            .map_err(|e| format!("找不到应用数据目录：{e}"))?,
    };
    let log_dir = data_dir.join("logs");
    fs::create_dir_all(&log_dir).map_err(|e| format!("无法创建数据目录：{e}"))?;
    cleanup_previous(&data_dir);
    let log_path = log_dir.join("server.log");
    let pid_path = log_dir.join("backend.pid");
    let log = File::create(&log_path).map_err(|e| format!("无法创建日志文件：{e}"))?;
    let log_err = log.try_clone().map_err(|e| e.to_string())?;

    let mut command = Command::new(&launch.program);
    command
        .args(&launch.args)
        .current_dir(&launch.cwd)
        .env("FACE_PORT", port.to_string())
        .env("FACE_DATA_DIR", &data_dir)
        .env("FACE_STATIC_DIR", &launch.static_dir)
        .env("FACE_MODELS_DIR", &launch.models_dir)
        .env("FACE_SHUTDOWN_TOKEN", token)
        // 桌面应用必须是干净的首启体验：不读取仓库里的照片或 v1 模板，人物一律由用户在后台自建
        .env("SKIP_MIGRATE", "1");
    #[cfg(debug_assertions)]
    command.env("FACE_LOG_REQUESTS", "1");

    command.stdout(Stdio::from(log)).stderr(Stdio::from(log_err));
    enter_process_group(&mut command);

    let child = command
        .spawn()
        .map_err(|e| format!("无法启动识别服务（{}）：{e}", launch.program.display()))?;
    let _ = fs::write(&pid_path, child.id().to_string());
    Ok((child, log_path, pid_path))
}

fn wait_until_ready(app: AppHandle, backend: Backend) {
    let Some(window) = app.get_webview_window("main") else { return };
    let started = Instant::now();
    let mut announced = false;
    let mut guard_created = false;

    loop {
        std::thread::sleep(Duration::from_millis(250));

        let exited = {
            let mut guard = backend.child.lock().unwrap();
            match guard.as_mut() {
                Some(child) => match child.try_wait() {
                    Ok(Some(status)) => Some(format!("识别服务已退出（{status}）")),
                    Ok(None) => None,
                    Err(e) => Some(format!("无法获取识别服务状态：{e}")),
                },
                None => Some("识别服务没有运行".to_string()),
            }
        };
        if let Some(message) = exited {
            let _ = window.eval(&format!(
                "window.__backendFailed && window.__backendFailed({});",
                serde_json::to_string(&message).unwrap_or_default()
            ));
            let _ = window.eval(&format!(
                "window.__setLogPath && window.__setLogPath({});",
                serde_json::to_string(&backend.log_path.display().to_string()).unwrap_or_default()
            ));
            return;
        }

        if health_ok(backend.port) {
            if !guard_created {
                guard_created = true;
                let handle = app.clone();
                let port = backend.port;
                let _ = app.run_on_main_thread(move || create_guard_window(&handle, port));
            }
            if !announced {
                announced = true;
                let _ = window.eval(&format!("window.__backendReady && window.__backendReady({});", backend.port));
            }
            // 启动页可能在 eval 之后才完成加载，重复几次导航是幂等的
            if started.elapsed() > Duration::from_secs(6) {
                return;
            }
        } else if started.elapsed() > Duration::from_secs(90) {
            let message = "等待识别服务超时（90 秒）".to_string();
            let _ = window.eval(&format!(
                "window.__backendFailed && window.__backendFailed({});",
                serde_json::to_string(&message).unwrap_or_default()
            ));
            let _ = window.eval(&format!(
                "window.__setLogPath && window.__setLogPath({});",
                serde_json::to_string(&backend.log_path.display().to_string()).unwrap_or_default()
            ));
            return;
        }
    }
}

fn stop_backend(app: &AppHandle) {
    let Some(backend) = app.try_state::<Backend>() else { return };
    let Some(mut child) = backend.child.lock().unwrap().take() else { return };

    request_shutdown(backend.port, &backend.token);
    let deadline = Instant::now() + Duration::from_secs(3);
    while Instant::now() < deadline {
        if let Ok(Some(_)) = child.try_wait() {
            let _ = fs::remove_file(&backend.pid_path);
            return;
        }
        std::thread::sleep(Duration::from_millis(100));
    }

    #[cfg(unix)]
    unsafe {
        libc::kill(-(child.id() as i32), libc::SIGTERM);
    }
    let _ = child.kill();
    let _ = child.wait();
    let _ = fs::remove_file(&backend.pid_path);
}

#[cfg(target_os = "macos")]
fn configure_guard_window(window: &tauri::WebviewWindow) {
    let Ok(ptr) = window.ns_window() else { return };
    unsafe {
        let ns = &*(ptr as *const NSWindow);
        ns.setLevel(NSScreenSaverWindowLevel);
        ns.setCollectionBehavior(
            NSWindowCollectionBehavior::CanJoinAllSpaces
                | NSWindowCollectionBehavior::FullScreenAuxiliary
                | NSWindowCollectionBehavior::Stationary
                | NSWindowCollectionBehavior::IgnoresCycle,
        );
        ns.setIgnoresMouseEvents(true);
        ns.setHidesOnDeactivate(false);
        ns.setOpaque(false);
        let clear = NSColor::clearColor();
        ns.setBackgroundColor(Some(&clear));
        ns.orderFrontRegardless();
    }
}

fn create_guard_window(app: &AppHandle, port: u16) {
    if app.get_webview_window("guard").is_some() {
        return;
    }

    let (width, height) = app
        .primary_monitor()
        .ok()
        .flatten()
        .map(|monitor| {
            let size = monitor.size().to_logical::<f64>(monitor.scale_factor());
            (size.width, size.height)
        })
        .unwrap_or((1440.0, 900.0));

    let url: tauri::Url = match format!("http://127.0.0.1:{port}/guard.html").parse() {
        Ok(url) => url,
        Err(e) => {
            eprintln!("距离提醒页面地址无效：{e}");
            return;
        }
    };

    let built = tauri::WebviewWindowBuilder::new(app, "guard", tauri::WebviewUrl::External(url))
        .title("距离提醒")
        .decorations(false)
        .transparent(true)
        .always_on_top(true)
        .skip_taskbar(true)
        .shadow(false)
        .resizable(false)
        .focused(false)
        .visible(false)
        .inner_size(width, height)
        .position(0.0, 0.0)
        .build();

    match built {
        Ok(window) => {
            #[cfg(target_os = "macos")]
            configure_guard_window(&window);
        }
        Err(e) => eprintln!("创建距离提醒窗口失败：{e}"),
    }
}

fn setup_tray(app: &tauri::App) -> tauri::Result<()> {
    let port = app.try_state::<Backend>().map(|b| b.port).unwrap_or(0);
    let toggle = CheckMenuItem::with_id(
        app,
        "guard-toggle",
        "距离提醒",
        true,
        guard_enabled_of(port).unwrap_or(false),
        None::<&str>,
    )?;
    let separator1 = tauri::menu::PredefinedMenuItem::separator(app)?;
    let test = MenuItem::with_id(app, "guard-test", "测试全屏警告", true, None::<&str>)?;
    let admin = MenuItem::with_id(app, "guard-admin", "打开管理后台", true, None::<&str>)?;
    let show = MenuItem::with_id(app, "guard-show", "显示主窗口", true, None::<&str>)?;
    let debug = MenuItem::with_id(app, "guard-debug", "距离提醒调试信息", true, None::<&str>)?;
    let separator2 = tauri::menu::PredefinedMenuItem::separator(app)?;
    let quit = MenuItem::with_id(app, "guard-quit", "退出", true, None::<&str>)?;
    let menu = Menu::with_items(app, &[&toggle, &separator1, &test, &admin, &show, &debug, &separator2, &quit])?;

    let sync_item = toggle.clone();
    let tray_item = toggle.clone();
    let mut tray = TrayIconBuilder::new()
        .menu(&menu)
        .show_menu_on_left_click(true)
        .tooltip("LocalFace · 距离提醒")
        .on_menu_event(move |app, event| match event.id.as_ref() {
            "guard-toggle" => {
                let Some(backend) = app.try_state::<Backend>() else { return };
                // 以菜单项当前勾选为切换意图：立即翻转给出反馈，后台再发请求并回读校准
                let next = !sync_item.is_checked().unwrap_or(false);
                let _ = sync_item.set_checked(next);
                let port = backend.port;
                let item = sync_item.clone();
                let handle = app.clone();
                std::thread::spawn(move || {
                    let body = if next { "{\"enabled\":true}" } else { "{\"enabled\":false}" };
                    let _ = http_request(port, "PUT", "/api/guard", Some(body));
                    let actual = guard_enabled_of(port).unwrap_or(next);
                    let _ = handle.run_on_main_thread(move || {
                        let _ = item.set_checked(actual);
                    });
                });
            }
            "guard-test" => {
                if let Some(window) = app.get_webview_window("guard") {
                    let _ = window.eval("window.__guardTest && window.__guardTest();");
                }
            }
            "guard-admin" => {
                if let Some(window) = app.get_webview_window("main") {
                    let _ = window.show();
                    let _ = window.unminimize();
                    let _ = window.set_focus();
                    if let Some(backend) = app.try_state::<Backend>() {
                        let port = backend.port;
                        let js = format!(
                            "if (location.protocol === 'http:' || location.protocol === 'https:') {{
                                location.replace('http://127.0.0.1:{port}/admin');
                            }} else {{
                                const go = () => fetch('http://127.0.0.1:{port}/api/state')
                                    .then(() => location.replace('http://127.0.0.1:{port}/admin'))
                                    .catch(() => setTimeout(go, 500));
                                go();
                            }}"
                        );
                        let _ = window.eval(&js);
                    }
                }
            }
            "guard-show" => {
                if let Some(window) = app.get_webview_window("main") {
                    let _ = window.show();
                    let _ = window.unminimize();
                    let _ = window.set_focus();
                }
            }
            "guard-debug" => {
                if let Some(window) = app.get_webview_window("main") {
                    let _ = window.show();
                    let _ = window.unminimize();
                    let _ = window.set_focus();
                    if let Some(backend) = app.try_state::<Backend>() {
                        let port = backend.port;
                        let js = format!(
                            "if (location.protocol === 'http:' || location.protocol === 'https:') {{
                                location.replace('http://127.0.0.1:{port}/guard?test=1');
                            }} else {{
                                const go = () => fetch('http://127.0.0.1:{port}/api/state')
                                    .then(() => location.replace('http://127.0.0.1:{port}/guard?test=1'))
                                    .catch(() => setTimeout(go, 500));
                                go();
                            }}"
                        );
                        let _ = window.eval(&js);
                    }
                }
            }
            "guard-quit" => app.exit(0),
            _ => {}
        });

    if let Some(icon) = app.default_window_icon() {
        tray = tray.icon(icon.clone());
    }
    tray.build(app)?;

    // 管理后台里也能改总开关，定时把托盘勾选状态对齐
    let sync_app = app.handle().clone();
    std::thread::spawn(move || loop {
        std::thread::sleep(Duration::from_secs(4));
        let Some(backend) = sync_app.try_state::<Backend>() else { continue };
        let Some(enabled) = guard_enabled_of(backend.port) else { continue };
        let item = tray_item.clone();
        let _ = sync_app.run_on_main_thread(move || {
            let _ = item.set_checked(enabled);
        });
    });
    Ok(())
}

pub fn run() {
    #[cfg(unix)]
    install_signal_handlers();

    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            camera_permission_status,
            request_camera_permission,
            open_camera_settings
        ])
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| {
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.unminimize();
                let _ = window.show();
                let _ = window.set_focus();
            }
        }))
        .setup(|app| {
            let handle = app.handle().clone();
            let port = free_port();
            let token = make_token();
            let launch = resolve_launch(&handle)?;
            let (child, log_path, pid_path) = start_backend(&handle, launch, port, &token)?;
            let backend = Backend {
                child: Arc::new(Mutex::new(Some(child))),
                port,
                token,
                log_path,
                pid_path,
            };
            app.manage(backend.clone());
            setup_tray(app)?;
            std::thread::spawn(move || wait_until_ready(handle, backend));

            #[cfg(unix)]
            {
                let handle = app.handle().clone();
                std::thread::spawn(move || loop {
                    if SHUTDOWN.load(Ordering::SeqCst) {
                        stop_backend(&handle);
                        std::process::exit(0);
                    }
                    std::thread::sleep(Duration::from_millis(150));
                });
            }
            Ok(())
        })
        .on_window_event(|window, event| {
            if window.label() == "main" {
                if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                    api.prevent_close();
                    let _ = window.hide();
                }
            }
        })
        .build(tauri::generate_context!())
        .expect("failed to build tauri application")
        .run(|app, event| {
            if let RunEvent::Exit = event {
                stop_backend(app);
            }
        });
}
