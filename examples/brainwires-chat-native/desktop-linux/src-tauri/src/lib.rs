use brainwires_chat_native_bridge as bridge;

#[tauri::command]
fn framework_version() -> String {
    bridge::framework_version()
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![framework_version])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
