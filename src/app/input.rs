//! --------------------------------------------------------------------------------------
//! Input Handling (input.rs)
//!
//! Created: August 2025
//! Author: Stephen Willey (with the AIs doing a bunch of the work and trying to teach me)
//!
//! This module provides basic input tracking and event handling for the application.
//!
//! --------------------------------------------------------------------------------------

use std::time::Instant;

use winit::event::{DeviceEvent, ElementState::{Pressed, Released}, Event, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::PhysicalKey::Code;
use winit::keyboard::KeyCode;
use winit::window::WindowId;

use crate::app::app::App;

#[derive(Default, Clone, Copy)]
pub struct InputState {
    pub moving_forward: bool,
    pub moving_backward: bool,
    pub moving_left: bool,
    pub moving_right: bool,
    pub toggle_locked: bool,
    pub mouselook_enabled: bool,
}

pub fn handle_device_event(app: &mut App, event: DeviceEvent) {
    if let DeviceEvent::MouseMotion { delta } = event {
        if app.input.mouselook_enabled {
            let (dx, dy) = (delta.0 as f32, delta.1 as f32);
            let sensitivity = 0.1;
            app.camera.rotate(dx * sensitivity, -dy * sensitivity);
        }
    }
}

pub fn handle_window_event(
    app: &mut App,
    event_loop: &ActiveEventLoop,
    window_id: WindowId,
    event: WindowEvent,
) {
    if let Some(win) = &app.window {
        if let (Some(platform), Some(imgui)) = (app.platform.as_mut(), app.imgui.as_mut()) {
            let full_event: Event<()> = Event::WindowEvent {
                window_id: win.id(),
                event: event.clone(),
            };
            platform.handle_event(imgui.io_mut(), win, &full_event);
        }
        if window_id == win.id() {
            match event {
                WindowEvent::CloseRequested => {
                    println!("Close requested!");
                    event_loop.exit();
                }
                WindowEvent::Resized(_) | WindowEvent::ScaleFactorChanged { .. } => {
                    println!("📐 Window resized");
                    if let Some(vulkan_base) = &mut app.vulkan_base {
                        if let Err(e) = vulkan_base.recreate_swapchain(app.window.as_ref().unwrap()) {
                            eprintln!("Failed to recreate swapchain: {}", e);
                        }
                    }
                    if let Some(window) = &app.window {
                        let size = window.inner_size();
                        let aspect = size.width as f32 / size.height as f32;
                        app.camera.set_perspective_projection(45.0, aspect, 0.1, 100.0);
                    }
                }
                WindowEvent::ModifiersChanged(mods) => {
                    app.modifiers = mods.state();
                }
                WindowEvent::KeyboardInput { event, .. } => {
                    let key = event.physical_key;
                    match (key, event.state) {
                        (Code(KeyCode::Escape), Pressed) => {
                            println!("🛑 Escape pressed, exiting");
                            event_loop.exit();
                        }
                        (Code(KeyCode::KeyF), Pressed) if !app.modifiers.control_key() => {
                            let fullscreen = if win.fullscreen().is_some() {
                                None
                            } else {
                                Some(winit::window::Fullscreen::Borderless(None))
                            };
                            win.set_fullscreen(fullscreen);
                            println!("🖥️ Toggled fullscreen");
                        }
                        (Code(KeyCode::KeyU), Released) => {
                            app.input.toggle_locked = false;
                        }
                        (Code(KeyCode::KeyW), Pressed) if !app.modifiers.control_key() => {
                            app.input.moving_forward = true;
                        }
                        (Code(KeyCode::KeyW), Released) => {
                            app.input.moving_forward = false;
                            app.input.toggle_locked = false;
                        }
                        (Code(KeyCode::KeyS), Pressed) => {
                            app.input.moving_backward = true;
                        }
                        (Code(KeyCode::KeyS), Released) => {
                            app.input.moving_backward = false;
                        }
                        (Code(KeyCode::KeyA), Pressed) => {
                            app.input.moving_left = true;
                        }
                        (Code(KeyCode::KeyA), Released) => {
                            app.input.moving_left = false;
                        }
                        (Code(KeyCode::KeyD), Pressed) => {
                            app.input.moving_right = true;
                        }
                        (Code(KeyCode::KeyD), Released) => {
                            app.input.moving_right = false;
                        }
                        (Code(KeyCode::KeyM), Pressed) if !app.input.toggle_locked => {
                            app.input.toggle_locked = true;
                            app.input.mouselook_enabled = !app.input.mouselook_enabled;
                        }
                        (Code(KeyCode::KeyM), Released) => {
                            app.input.toggle_locked = false;
                        }
                        (Code(KeyCode::KeyU), Pressed)
                            if !app.input.toggle_locked =>
                        {
                            app.input.toggle_locked = true;
                            if let Some(vb) = &mut app.vulkan_base {
                                vb.toggle_ui();
                            }
                            let show = if let Some(vb) = &app.vulkan_base {
                                vb.engine_settings.show_ui
                            } else {
                                false
                            };
                            app.render_graph.set_pass_enabled(crate::vulkan::render_graph::RenderPassNode::UI, show);
                        }
                        _ => {}
                    }
                }
                WindowEvent::RedrawRequested => {
                    // Exit requested via UI? Signal the event loop to exit.
                    if app.exit_flag.load(std::sync::atomic::Ordering::Relaxed) {
                        println!("👋 Exit requested from UI");
                        event_loop.exit();
                        return;
                    }

                    app.start_of_frame_time = Instant::now();

                    let s = app.step;
                    if app.input.moving_forward {
                        app.camera.translate(s, 0.0)
                    }
                    if app.input.moving_backward {
                        app.camera.translate(-s, 0.0)
                    }
                    if app.input.moving_left {
                        app.camera.translate(0.0, -s)
                    }
                    if app.input.moving_right {
                        app.camera.translate(0.0, s)
                    }

                    let mut graph = std::mem::take(&mut app.render_graph);
                    if let Err(e) = graph.execute(app) {
                        eprintln!("draw_frame error: {}", e);
                    }
                    app.render_graph = graph;

                    let now = Instant::now();
                    let elapsed = now.duration_since(app.start_of_frame_time);
                    app.current_ms_per_frame = elapsed.as_millis() as f32;
                }
                _ => {}
            }
        }
    }
}
