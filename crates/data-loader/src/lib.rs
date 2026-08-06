use datafusion::{
    arrow::{
        array::{Array, AsArray, Float32Array},
        datatypes::Float32Type,
    },
    error::DataFusionError,
    execution::{context::SessionContext, options::ParquetReadOptions},
};

pub async fn load_embeddings(path: &str) -> Result<Vec<Vec<f32>>, DataFusionError> {
    let ctx = SessionContext::new();
    ctx.register_parquet("data", path, ParquetReadOptions::default())
        .await?;
    let df = ctx.sql("select emb from data").await?;
    let batches = df.collect().await?;
    let mut embeddings: Vec<Vec<f32>> = vec![];
    for batch in &batches {
        let column = batch.column(0);
        let list_array = column.as_fixed_size_list();

        for i in 0..list_array.len() {
            if list_array.is_null(i) {
                continue;
            }
            let values = list_array.value(i);
            let float_array: &Float32Array = values.as_primitive::<Float32Type>();
            let vec: Vec<f32> = float_array.values().to_vec();
            embeddings.push(vec);
        }
    }

    Ok(embeddings)
}
