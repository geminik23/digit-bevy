use bevy::{
    asset::{io::Reader, AssetLoader, AsyncReadExt, LoadContext},
    prelude::*,
    reflect::TypeUuid,
};

#[derive(Asset, TypeUuid, TypePath)]
#[uuid = "1f3fcf03-af83-4896-b345-f12a64beb8a3"]
pub struct WONNXModel {
    pub model: wonnx::Session,
}

#[derive(Default)]
pub struct WONNXLoader;

impl AssetLoader for WONNXLoader {
    type Asset = WONNXModel;
    type Settings = ();
    type Error = anyhow::Error;

    fn load<'a>(
        &'a self,
        reader: &'a mut Reader,
        _settings: &'a Self::Settings,
        _load_context: &'a mut LoadContext,
    ) -> bevy::utils::BoxedFuture<'a, Result<Self::Asset, Self::Error>> {
        Box::pin(async move {
            let mut bytes = vec![];
            let _ = reader.read_to_end(&mut bytes).await;

            Ok(WONNXModel {
                model: wonnx::Session::from_bytes(&bytes).await?,
            })
        })
    }

    fn extensions(&self) -> &[&str] {
        &["onnx"]
    }
}
