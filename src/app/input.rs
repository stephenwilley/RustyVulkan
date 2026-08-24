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

use crate::app::app::{
    App, FIRST_PERSON_EYE_HEIGHT, capture_first_person_cursor, release_first_person_cursor,
};

/// Tracks first-person movement and whether the pointer is released for UI interaction.
#[derive(Default, Clone, Copy)]
pub struct InputState {
    pub moving_forward: bool,
    pub moving_backward: bool,
    pub moving_left: bool,
    pub moving_right: bool,
    /// `true` while ImGui owns a visible, freely moving pointer.
    pub cursor_released: bool,
    /// Prevents one physical U press from toggling repeatedly through key repeat events.
    pub cursor_toggle_locked: bool,
}

/// Handles raw relative motion only while the cursor is captured for first-person look.
pub fn handle_device_event(app: &mut App, event: DeviceEvent) {
    if !app.input.cursor_released
        && let DeviceEvent::MouseMotion { delta } = event
    {
        let (dx, dy) = (delta.0 as f32, delta.1 as f32);
        let sensitivity = 0.1;
        app.camera.rotate(dx * sensitivity, -dy * sensitivity);
    }
}

/// Handles window events and routes them to ImGui, window management, movement, and rendering.
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
                    let size = win.inner_size();
                    // A minimized window has a zero-sized surface.  Vulkan cannot create
                    // a swapchain for it, and width / height would produce an invalid
                    // camera aspect ratio.  The next non-zero resize will recreate it.
                    if size.width == 0 || size.height == 0 {
                        return;
                    }
                    if let Some(vulkan_base) = &mut app.vulkan_base {
                        if let Err(e) = vulkan_base.recreate_swapchain(app.window.as_ref().unwrap()) {
                            eprintln!("Failed to recreate swapchain: {}", e);
                        }
                    }
                    let aspect = size.width as f32 / size.height as f32;
                    // Preserve current fov/near/far; adjust only aspect on resize.
                    app.camera.set_aspect(aspect);
                }
                WindowEvent::Focused(false) => {
                    // Winit does not guarantee key-release events while the window is
                    // unfocused; clear movement so the camera cannot get "stuck".
                    app.input.moving_forward = false;
                    app.input.moving_backward = false;
                    app.input.moving_left = false;
                    app.input.moving_right = false;
                    app.input.cursor_toggle_locked = false;
                    app.last_movement_update = Instant::now();
                }
                WindowEvent::Focused(true) if !app.input.cursor_released => {
                    // Some window systems release the grab when focus is lost.
                    capture_first_person_cursor(win);
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
                        (Code(KeyCode::KeyW), Pressed) if !app.modifiers.control_key() => {
                            app.input.moving_forward = true;
                        }
                        (Code(KeyCode::KeyW), Released) => {
                            app.input.moving_forward = false;
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
                        (Code(KeyCode::KeyU), Pressed)
                            if !app.input.cursor_toggle_locked =>
                        {
                            app.input.cursor_toggle_locked = true;
                            app.input.cursor_released = !app.input.cursor_released;
                            if app.input.cursor_released {
                                release_first_person_cursor(win);
                            } else {
                                capture_first_person_cursor(win);
                            }
                        }
                        (Code(KeyCode::KeyU), Released) => {
                            app.input.cursor_toggle_locked = false;
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

                    let now = Instant::now();
                    // Clamp long pauses (breakpoint, drag, focus loss) so resuming cannot
                    // make the camera jump through the scene in a single frame.
                    let delta_seconds = now
                        .duration_since(app.last_movement_update)
                        .as_secs_f32()
                        .min(0.1);
                    app.last_movement_update = now;

                    let mut forward_input = 0.0_f32;
                    let mut right_input = 0.0_f32;
                    if app.input.moving_forward { forward_input += 1.0; }
                    if app.input.moving_backward { forward_input -= 1.0; }
                    if app.input.moving_right { right_input += 1.0; }
                    if app.input.moving_left { right_input -= 1.0; }

                    // Normalize WASD intent: W+D covers the same metres/second as W alone.
                    let input_length = (forward_input * forward_input + right_input * right_input).sqrt();
                    if input_length > 0.0 {
                        let distance = app.walk_speed_mps * delta_seconds / input_length;
                        app.camera.translate_horizontal(
                            forward_input * distance,
                            right_input * distance,
                        );
                    }

                    // The terrain function is shared with mesh generation, so the eye is
                    // always exactly 1.8 m above the surface the player sees.
                    let camera_position = app.camera.position();
                    app.camera.set_height_above_ground(
                        app.terrain_settings
                            .height_at(camera_position.x, camera_position.z),
                        FIRST_PERSON_EYE_HEIGHT,
                    );

                    let mut graph = std::mem::take(&mut app.render_graph);
                    if let Err(e) = graph.execute(app) {
                        eprintln!("draw_frame error: {}", e);
                    }
                    app.render_graph = graph;

                    // CPU recording time is measured inside the render graph now.
                }
                _ => {}
            }
        }
    }
}
