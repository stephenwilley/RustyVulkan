mod app;
mod vulkan;
mod graphics;
mod camera;
mod scene;

use app::App;

fn main() {
    if let Err(e) = App::new().run() {
        eprintln!("Application error: {}", e);
    }
}

