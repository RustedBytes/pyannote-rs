// Generated from ONNX "segmentation-3.0.onnx" by burn-import
use burn::prelude::*;
use burn::nn::BiLstm;
use burn::nn::BiLstmConfig;
use burn::nn::InstanceNorm;
use burn::nn::InstanceNormConfig;
use burn::nn::Linear;
use burn::nn::LinearConfig;
use burn::nn::LstmState;
use burn::nn::PaddingConfig1d;
use burn::nn::conv::Conv1d;
use burn::nn::conv::Conv1dConfig;
use burn::nn::pool::MaxPool1d;
use burn::nn::pool::MaxPool1dConfig;
use burn::tensor::activation::log_softmax;
use burn_store::BurnpackStore;
use burn_store::ModuleSnapshot;


#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    instancenormalization1: InstanceNorm<B>,
    conv1d7: Conv1d<B>,
    conv1d8: Conv1d<B>,
    maxpool1d1: MaxPool1d,
    instancenormalization2: InstanceNorm<B>,
    conv1d1: Conv1d<B>,
    maxpool1d2: MaxPool1d,
    instancenormalization3: InstanceNorm<B>,
    conv1d2: Conv1d<B>,
    maxpool1d3: MaxPool1d,
    instancenormalization4: InstanceNorm<B>,
    lstm1: BiLstm<B>,
    lstm2: BiLstm<B>,
    lstm3: BiLstm<B>,
    lstm4: BiLstm<B>,
    linear1: Linear<B>,
    linear2: Linear<B>,
    linear3: Linear<B>,
    phantom: core::marker::PhantomData<B>,
    device: burn::module::Ignored<B::Device>,
}


impl<B: Backend> Default for Model<B> {
    fn default() -> Self {
        Self::from_file("nn/segmentation/model.bpk", &Default::default())
    }
}

impl<B: Backend> Model<B> {
    /// Load model weights from a burnpack file.
    pub fn from_file(file: &str, device: &B::Device) -> Self {
        let mut model = Self::new(device);
        let mut store = BurnpackStore::from_file(file);
        model.load_from(&mut store).expect("Failed to load burnpack file");
        model
    }
}

impl<B: Backend> Model<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let instancenormalization1 = InstanceNormConfig::new(1)
            .with_epsilon(0.000009999999747378752f64)
            .init(device);
        let conv1d7 = Conv1dConfig::new(1, 80, 251)
            .with_stride(10)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let conv1d8 = Conv1dConfig::new(1, 80, 251)
            .with_stride(10)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let maxpool1d1 = MaxPool1dConfig::new(3)
            .with_stride(3)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_ceil_mode(false)
            .init();
        let instancenormalization2 = InstanceNormConfig::new(80)
            .with_epsilon(0.000009999999747378752f64)
            .init(device);
        let conv1d1 = Conv1dConfig::new(80, 60, 5)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let maxpool1d2 = MaxPool1dConfig::new(3)
            .with_stride(3)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_ceil_mode(false)
            .init();
        let instancenormalization3 = InstanceNormConfig::new(60)
            .with_epsilon(0.000009999999747378752f64)
            .init(device);
        let conv1d2 = Conv1dConfig::new(60, 60, 5)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let maxpool1d3 = MaxPool1dConfig::new(3)
            .with_stride(3)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_ceil_mode(false)
            .init();
        let instancenormalization4 = InstanceNormConfig::new(60)
            .with_epsilon(0.000009999999747378752f64)
            .init(device);
        let lstm1 = BiLstmConfig::new(60, 128, true)
            .with_batch_first(false)
            .with_input_forget(false)
            .init(device);
        let lstm2 = BiLstmConfig::new(256, 128, true)
            .with_batch_first(false)
            .with_input_forget(false)
            .init(device);
        let lstm3 = BiLstmConfig::new(256, 128, true)
            .with_batch_first(false)
            .with_input_forget(false)
            .init(device);
        let lstm4 = BiLstmConfig::new(256, 128, true)
            .with_batch_first(false)
            .with_input_forget(false)
            .init(device);
        let linear1 = LinearConfig::new(256, 128).with_bias(true).init(device);
        let linear2 = LinearConfig::new(128, 128).with_bias(true).init(device);
        let linear3 = LinearConfig::new(128, 7).with_bias(true).init(device);
        Self {
            instancenormalization1,
            conv1d7,
            conv1d8,
            maxpool1d1,
            instancenormalization2,
            conv1d1,
            maxpool1d2,
            instancenormalization3,
            conv1d2,
            maxpool1d3,
            instancenormalization4,
            lstm1,
            lstm2,
            lstm3,
            lstm4,
            linear1,
            linear2,
            linear3,
            phantom: core::marker::PhantomData,
            device: burn::module::Ignored(device.clone()),
        }
    }

    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let constant4_out1 = 0i64;
        let constant6_out1 = 1i64;
        let constant39_out1: [i64; 1] = [128i64];
        let constant40_out1: [i64; 1] = [8i64];
        let constant41_out1: [i64; 3] = [0i64, 2i64, 1i64];
        let instancenormalization1_out1 = self.instancenormalization1.forward(input);
        let shape1_out1: [i64; 3] = {
            let axes = &instancenormalization1_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let actual_idx = if constant6_out1 < 0 {
            (shape1_out1.len() as i64 + constant6_out1) as usize
        } else {
            constant6_out1 as usize
        };
        let gather1_out1 = shape1_out1[actual_idx] as i64;
        let equal1_out1 = gather1_out1 == constant6_out1;
        let if1_out1 = if equal1_out1 {
            let sincnet_wav_norm1d_instance_normalization_output_0 = instancenormalization1_out1
                .clone();
            let conv1d7_out1 = self
                .conv1d7
                .forward(sincnet_wav_norm1d_instance_normalization_output_0);
            conv1d7_out1
        } else {
            let sincnet_wav_norm1d_instance_normalization_output_0 = instancenormalization1_out1
                .clone();
            let constant89_out1: [i64; 1] = [1i64];
            let constant99_out1: [i64; 1] = [-1i64];
            let slice10_out1: [i64; 1] = shape1_out1[2..3].try_into().unwrap();
            let squeeze4_out1 = slice10_out1[0] as i64;
            let unsqueeze4_out1 = [squeeze4_out1];
            let concat7_out1: [i64; 3usize] = [
                &constant99_out1[..],
                &constant89_out1[..],
                &unsqueeze4_out1[..],
            ]
                .concat()
                .try_into()
                .unwrap();
            let reshape7_out1 = sincnet_wav_norm1d_instance_normalization_output_0
                .reshape(concat7_out1);
            let conv1d8_out1 = self.conv1d8.forward(reshape7_out1);
            let shape6_out1: [i64; 3] = {
                let axes = &conv1d8_out1.clone().dims()[0..3];
                let mut output = [0i64; 3];
                for i in 0..3 {
                    output[i] = axes[i] as i64;
                }
                output
            };
            let slice11_out1: [i64; 2] = shape6_out1[1..3].try_into().unwrap();
            let slice12_out1: [i64; 2] = shape1_out1[0..2].try_into().unwrap();
            let concat8_out1: [i64; 4usize] = [&slice12_out1[..], &slice11_out1[..]]
                .concat()
                .try_into()
                .unwrap();
            let reshape8_out1 = conv1d8_out1.reshape([
                (concat8_out1[0] * concat8_out1[1]) as usize,
                concat8_out1[2] as usize,
                concat8_out1[3] as usize,
            ]);
            reshape8_out1
        };
        let abs1_out1 = if1_out1.abs();
        let maxpool1d1_out1 = self.maxpool1d1.forward(abs1_out1);
        let instancenormalization2_out1 = self
            .instancenormalization2
            .forward(maxpool1d1_out1);
        let leakyrelu1_out1 = burn::tensor::activation::leaky_relu(
            instancenormalization2_out1,
            0.009999999776482582,
        );
        let conv1d1_out1 = self.conv1d1.forward(leakyrelu1_out1);
        let maxpool1d2_out1 = self.maxpool1d2.forward(conv1d1_out1);
        let instancenormalization3_out1 = self
            .instancenormalization3
            .forward(maxpool1d2_out1);
        let leakyrelu2_out1 = burn::tensor::activation::leaky_relu(
            instancenormalization3_out1,
            0.009999999776482582,
        );
        let conv1d2_out1 = self.conv1d2.forward(leakyrelu2_out1);
        let maxpool1d3_out1 = self.maxpool1d3.forward(conv1d2_out1);
        let instancenormalization4_out1 = self
            .instancenormalization4
            .forward(maxpool1d3_out1);
        let leakyrelu3_out1 = burn::tensor::activation::leaky_relu(
            instancenormalization4_out1,
            0.009999999776482582,
        );
        let shape2_out1: [i64; 3] = {
            let axes = &leakyrelu3_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather2_out1: [i64; 3usize] = constant41_out1
            .iter()
            .map(|&idx| {
                let actual_idx = if idx < 0 {
                    (shape2_out1.len() as i64 + idx) as usize
                } else {
                    idx as usize
                };
                shape2_out1[actual_idx]
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let actual_idx = if constant4_out1 < 0 {
            (gather2_out1.len() as i64 + constant4_out1) as usize
        } else {
            constant4_out1 as usize
        };
        let gather3_out1 = gather2_out1[actual_idx] as i64;
        let unsqueeze1_out1 = [gather3_out1];
        let concat1_out1: [i64; 3usize] = [
            &constant40_out1[..],
            &unsqueeze1_out1[..],
            &constant39_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let constantofshape1_out1 = Tensor::<
            B,
            1,
        >::from_data_dtype(
                burn::tensor::TensorData::from([0f32 as f64]),
                &*self.device,
                burn::tensor::DType::F32,
            )
            .reshape([1, 1, 1])
            .expand(concat1_out1);
        let slice1_out1 = constantofshape1_out1.clone().slice(s![6..8, .., ..]);
        let slice2_out1 = constantofshape1_out1.clone().slice(s![4..6, .., ..]);
        let slice3_out1 = constantofshape1_out1.clone().slice(s![2..4, .., ..]);
        let slice4_out1 = constantofshape1_out1.slice(s![0..2, .., ..]);
        let transpose1_out1 = leakyrelu3_out1.permute([2, 0, 1]);
        let (lstm1_out1, _lstm1_out2, _lstm1_out3) = {
            let (output_seq, final_state) = self
                .lstm1
                .forward(
                    transpose1_out1,
                    Some(LstmState::new(slice4_out1.clone(), slice4_out1)),
                );
            (
                {
                    let [seq_len, batch_size, _] = output_seq.dims();
                    let reshaped = output_seq
                        .reshape([seq_len, batch_size, 2, 128usize]);
                    reshaped.swap_dims(1, 2)
                },
                final_state.hidden,
                final_state.cell,
            )
        };
        let transpose2_out1 = lstm1_out1.permute([0, 2, 1, 3]);
        let reshape1_out1 = transpose2_out1.reshape([0, 0, -1]);
        let (lstm2_out1, _lstm2_out2, _lstm2_out3) = {
            let (output_seq, final_state) = self
                .lstm2
                .forward(
                    reshape1_out1,
                    Some(LstmState::new(slice3_out1.clone(), slice3_out1)),
                );
            (
                {
                    let [seq_len, batch_size, _] = output_seq.dims();
                    let reshaped = output_seq
                        .reshape([seq_len, batch_size, 2, 128usize]);
                    reshaped.swap_dims(1, 2)
                },
                final_state.hidden,
                final_state.cell,
            )
        };
        let transpose3_out1 = lstm2_out1.permute([0, 2, 1, 3]);
        let reshape2_out1 = transpose3_out1.reshape([0, 0, -1]);
        let (lstm3_out1, _lstm3_out2, _lstm3_out3) = {
            let (output_seq, final_state) = self
                .lstm3
                .forward(
                    reshape2_out1,
                    Some(LstmState::new(slice2_out1.clone(), slice2_out1)),
                );
            (
                {
                    let [seq_len, batch_size, _] = output_seq.dims();
                    let reshaped = output_seq
                        .reshape([seq_len, batch_size, 2, 128usize]);
                    reshaped.swap_dims(1, 2)
                },
                final_state.hidden,
                final_state.cell,
            )
        };
        let transpose4_out1 = lstm3_out1.permute([0, 2, 1, 3]);
        let reshape3_out1 = transpose4_out1.reshape([0, 0, -1]);
        let (lstm4_out1, _lstm4_out2, _lstm4_out3) = {
            let (output_seq, final_state) = self
                .lstm4
                .forward(
                    reshape3_out1,
                    Some(LstmState::new(slice1_out1.clone(), slice1_out1)),
                );
            (
                {
                    let [seq_len, batch_size, _] = output_seq.dims();
                    let reshaped = output_seq
                        .reshape([seq_len, batch_size, 2, 128usize]);
                    reshaped.swap_dims(1, 2)
                },
                final_state.hidden,
                final_state.cell,
            )
        };
        let transpose5_out1 = lstm4_out1.permute([0, 2, 1, 3]);
        let reshape4_out1 = transpose5_out1.reshape([0, 0, -1]);
        let transpose6_out1 = reshape4_out1.permute([1, 0, 2]);
        let linear1_out1 = self.linear1.forward(transpose6_out1);
        let leakyrelu4_out1 = burn::tensor::activation::leaky_relu(
            linear1_out1,
            0.009999999776482582,
        );
        let linear2_out1 = self.linear2.forward(leakyrelu4_out1);
        let leakyrelu5_out1 = burn::tensor::activation::leaky_relu(
            linear2_out1,
            0.009999999776482582,
        );
        let linear3_out1 = self.linear3.forward(leakyrelu5_out1);
        let logsoftmax1_out1 = log_softmax(linear3_out1, 2);
        logsoftmax1_out1
    }
}
