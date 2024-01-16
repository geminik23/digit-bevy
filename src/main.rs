mod onnx_loader;
use onnx_loader::{WONNXLoader, WONNXModel};

use std::{cmp::Ordering, collections::HashMap};
use wonnx::utils::InputTensor;

use bevy::{
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
};
use image::{imageops::FilterType, DynamicImage, Pixel};

// constant
pub const WINDOW_SIZE: (f32, f32) = (480.0, 480.0);
pub const CANVAS_SIZE: (u32, u32) = (64, 64);

fn line_points(mut xy1: Vec2, mut xy2: Vec2) -> Vec<Vec2> {
    // check the slope
    let swap = (xy2.y - xy1.y).abs() > (xy2.x - xy1.x).abs();
    if swap {
        xy1 = xy1.yx();
        xy2 = xy2.yx();
    }

    if xy1.x > xy2.x {
        let temp = xy1;
        xy1 = xy2;
        xy2 = temp;
    }

    let mut result = Vec::new();

    let mut x = xy1.x;
    let mut y = xy1.y;
    let dx = xy2.x - xy1.x;
    let dy = xy2.y - xy1.y;
    let mut d = 2.0 * dy - dx; // discriminator

    while x <= xy2.x {
        result.push(if swap {
            Vec2::new(y, x)
        } else {
            Vec2::new(x, y)
        });

        result.push(if swap {
            Vec2::new(y + 1.0, x)
        } else {
            Vec2::new(x + 1.0, y)
        });

        result.push(if swap {
            Vec2::new(y, x + 1.0)
        } else {
            Vec2::new(x, y + 1.0)
        });

        result.push(if swap {
            Vec2::new(y - 1.0, x)
        } else {
            Vec2::new(x - 1.0, y)
        });

        result.push(if swap {
            Vec2::new(y, x - 1.0)
        } else {
            Vec2::new(x, y - 1.0)
        });

        x = x + 1.0;
        if d <= 0.0 {
            d = d + 2.0 * dy;
        } else {
            d = d + 2.0 * (dy - dx);
            y = y + 1.0;
        }
    }
    //
    result
}

#[derive(Event)]
enum CanvasEvent {
    DrawAt(Vec2, f32),
    Clear,
}

#[derive(Component)]
pub struct Canvas;

#[derive(Component)]
pub struct ResultValue;

fn setup(mut commands: Commands, asset_server: Res<AssetServer>, mut state: ResMut<State>) {
    state.handle = asset_server.load("mnist-model.onnx");
    // camera settings
    let camera = Camera2dBundle::default();
    commands.spawn(camera);

    // texture to draw
    let image = Image::new_fill(
        Extent3d {
            width: CANVAS_SIZE.0,
            height: CANVAS_SIZE.1,
            ..Default::default()
        },
        TextureDimension::D2,
        &[0, 0, 0, 255],
        TextureFormat::Rgba8UnormSrgb,
    );
    // info!("{}", image.data.len()); // output : width*height*4 (bytes)

    let image = asset_server.add(image);

    commands
        .spawn(NodeBundle {
            style: Style {
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                flex_direction: FlexDirection::Column,
                justify_content: JustifyContent::Center,
                ..default()
            },
            ..default()
        })
        .with_children(|parent| {
            parent
                .spawn(ImageBundle {
                    style: Style {
                        width: Val::Percent(100.0),
                        height: Val::Percent(100.0),
                        ..default()
                    },
                    image: UiImage::new(image),
                    ..default()
                })
                .insert(Canvas)
                .insert(Interaction::None);
        });

    // Text
    let text_style = TextStyle {
        font_size: 20.0,
        color: Color::WHITE,
        font: Default::default(),
    };

    commands.spawn((
        TextBundle::from_section("", text_style.clone()).with_style(Style {
            position_type: PositionType::Absolute,
            top: Val::Px(10.0),
            left: Val::Px(10.0),
            ..default()
        }),
        ResultValue,
    ));
}

fn on_canvas_event(
    mut event_reader: EventReader<CanvasEvent>,
    mut assets: ResMut<Assets<Image>>,
    mut canvas: Query<&UiImage, With<Canvas>>,
) {
    let image = canvas.get_single_mut().unwrap();
    let image = image.texture.clone();
    let image = assets.get_mut(image.id());
    if image.is_none() {
        return;
    }
    let image = image.unwrap();
    // image.data[

    for event in event_reader.read() {
        match event {
            CanvasEvent::DrawAt(pos, intensity) => {
                let intensity = if *intensity > 1.0 { 1.0 } else { *intensity };

                if pos.x < 0.0 || pos.x >= 1.0 {
                    return;
                }
                if pos.y < 0.0 || pos.y >= 1.0 {
                    return;
                }
                let x = (CANVAS_SIZE.0 as f32 * pos.x + 0.5) as u32;
                let y = (CANVAS_SIZE.1 as f32 * pos.y + 0.5) as u32;
                let x = if x > CANVAS_SIZE.0 - 1 {
                    CANVAS_SIZE.0
                } else {
                    x
                };
                let y = if y > CANVAS_SIZE.1 - 1 {
                    CANVAS_SIZE.1
                } else {
                    y
                };

                let offset = (y * CANVAS_SIZE.1 + x) as usize;
                let val = (255.0 * intensity) as u8;
                image.data[offset * 4] = val;
                image.data[offset * 4 + 1] = val;
                image.data[offset * 4 + 2] = val;
                image.data[offset * 4 + 3] = 255;
            }
            CanvasEvent::Clear => {
                for i in 0..image.data.len() / 4 {
                    image.data[i * 4] = 0;
                    image.data[i * 4 + 1] = 0;
                    image.data[i * 4 + 2] = 0;
                    image.data[i * 4 + 3] = 255;
                }
            }
        }
    }
}

fn draw_on_mouse_move(
    mut cursor_reader: EventReader<CursorMoved>,
    mut last_point: Local<Option<Vec2>>,
    mouse_button: Res<Input<MouseButton>>,
    mut event_writer: EventWriter<CanvasEvent>,
    canvas: Query<(&Node, &Interaction, &GlobalTransform), With<Canvas>>,
) {
    // Style for size, Interaction to detect drawing, GlobalTransform for global position.
    //
    if mouse_button.just_released(MouseButton::Left) {
        *last_point = None;
        // send the 'Clear' event
        event_writer.send(CanvasEvent::Clear);
    }
    for (node, interaction, transform) in canvas.iter() {
        match interaction {
            Interaction::Pressed => {
                for cursor in cursor_reader.read() {
                    let size = node.size();
                    let xy = cursor.position.xy();
                    let trans = transform.translation().xy();

                    // info!("Node size : {:?}", size);
                    // info!("Global Translation : {:?}", trans);
                    // info!("Move position : {:?}", xy);

                    let mut points = vec![];

                    if let Some(last_point) = *last_point {
                        points = line_points(last_point, xy);
                    }
                    *last_point = Some(Vec2::new(xy.x, xy.y));

                    // normalized position [0-1]
                    for xy in points.into_iter() {
                        let norm_x = (xy.x - trans.x + size.x / 2.0) / size.x;
                        let norm_y = (xy.y - trans.y + size.y / 2.0) / size.y;

                        event_writer.send(CanvasEvent::DrawAt(Vec2::new(norm_x, norm_y), 1.0));
                    }
                }
            }
            _ => {}
        }
    }
    //
}

use pollster::FutureExt as _;
fn inference(
    state: Res<State>,
    mut img_assets: ResMut<Assets<Image>>,
    mut assets: ResMut<Assets<WONNXModel>>,
    canvas: Query<&UiImage, With<Canvas>>,
    mut display: Query<&mut Text, With<ResultValue>>,
) {
    let img = canvas.single();
    let img = img.texture.clone();
    let img = img_assets.get_mut(img.id());

    let Some(img) = img else { return };
    let Some(model) = assets.get_mut(state.handle.id()) else {
        return;
    };

    let mut input: HashMap<String, InputTensor<'_>> = HashMap::new();
    // rectangular
    let size = img.texture_descriptor.size;

    //
    let img_buffer = DynamicImage::ImageRgba8(
        image::RgbaImage::from_raw(size.width, size.height, img.data.clone()).unwrap(),
    );
    let img_buffer = img_buffer.resize(28, 28, FilterType::Triangle).to_luma8();

    let img = ndarray::Array::from_shape_fn((1, 1, 28, 28), |(_, _, j, i)| {
        // get pixel at ...
        let val = img_buffer.get_pixel(i as u32, j as u32).0;
        // [((i as f64 * delta) as usize + (j * img.texture_descriptor.size.width as usize)) * 4];
        if val[0] == 0 {
            0.0
        } else {
            1.0
        }
    });

    //
    let mut fin_value = None;

    if img.sum() == 0.0 {
        fin_value = Some("".to_string());
    } else {
        input.insert("Input3".to_string(), img.as_slice().unwrap().into());
        match model.model.run(&input).block_on() {
            Ok(output) => {
                //
                let (_, logits) = output.into_iter().next().unwrap();
                let logits: Vec<f32> = logits.try_into().unwrap();
                let result = logits
                    .into_iter()
                    .enumerate()
                    .max_by(|x, y| {
                        if x.1 > y.1 {
                            Ordering::Greater
                        } else {
                            Ordering::Less
                        }
                    })
                    .unwrap();
                fin_value = Some(format!("{}", result.0));
                // assert_eq!(result.0, 5);
                // log::info!("Result digit is {}", result.0);
            }
            Err(err) => {
                // log::error!("Run model error - {:?}", err);
            }
        }
    }

    // update the text
    let mut text = display.single_mut();
    let Some(fin_value) = fin_value else { return };
    text.sections[0].value = fin_value;
}

#[derive(Resource, Default)]
struct State {
    handle: Handle<WONNXModel>,
}

fn main() {
    App::new()
        .add_plugins((DefaultPlugins
            .set(WindowPlugin {
                primary_window: Some(Window {
                    title: "Drawing on canvas".into(),
                    resolution: WINDOW_SIZE.into(),
                    resizable: false,
                    ..Default::default()
                }),
                ..default()
            })
            .set(ImagePlugin::default_nearest()),))
        .init_resource::<State>()
        .init_asset::<WONNXModel>()
        .init_asset_loader::<WONNXLoader>()
        .add_event::<CanvasEvent>()
        .add_systems(Startup, setup)
        .add_systems(Update, (on_canvas_event, draw_on_mouse_move, inference))
        .run();
}
