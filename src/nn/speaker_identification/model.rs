// Generated from ONNX "wespeaker_final.onnx" by burn-import
use burn::prelude::*;
use burn::nn::BatchNorm;
use burn::nn::BatchNormConfig;
use burn::nn::PaddingConfig1d;
use burn::nn::PaddingConfig2d;
use burn::nn::conv::Conv1d;
use burn::nn::conv::Conv1dConfig;
use burn::nn::conv::Conv2d;
use burn::nn::conv::Conv2dConfig;
use burn::nn::pool::AvgPool1d;
use burn::nn::pool::AvgPool1dConfig;
use burn_store::BurnpackStore;
use burn_store::ModuleSnapshot;


#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    conv2d1: Conv2d<B>,
    conv2d2: Conv2d<B>,
    conv2d3: Conv2d<B>,
    conv2d4: Conv2d<B>,
    conv2d5: Conv2d<B>,
    conv2d6: Conv2d<B>,
    conv2d7: Conv2d<B>,
    conv2d8: Conv2d<B>,
    conv2d9: Conv2d<B>,
    conv2d10: Conv2d<B>,
    conv2d11: Conv2d<B>,
    conv2d12: Conv2d<B>,
    conv1d1: Conv1d<B>,
    batchnormalization1: BatchNorm<B>,
    conv1d2: Conv1d<B>,
    conv1d3: Conv1d<B>,
    averagepool1d1: AvgPool1d,
    conv1d4: Conv1d<B>,
    conv1d5: Conv1d<B>,
    batchnormalization2: BatchNorm<B>,
    conv1d6: Conv1d<B>,
    conv1d7: Conv1d<B>,
    averagepool1d2: AvgPool1d,
    conv1d8: Conv1d<B>,
    conv1d9: Conv1d<B>,
    batchnormalization3: BatchNorm<B>,
    conv1d10: Conv1d<B>,
    conv1d11: Conv1d<B>,
    averagepool1d3: AvgPool1d,
    conv1d12: Conv1d<B>,
    conv1d13: Conv1d<B>,
    batchnormalization4: BatchNorm<B>,
    conv1d14: Conv1d<B>,
    conv1d15: Conv1d<B>,
    averagepool1d4: AvgPool1d,
    conv1d16: Conv1d<B>,
    conv1d17: Conv1d<B>,
    batchnormalization5: BatchNorm<B>,
    conv1d18: Conv1d<B>,
    conv1d19: Conv1d<B>,
    averagepool1d5: AvgPool1d,
    conv1d20: Conv1d<B>,
    conv1d21: Conv1d<B>,
    batchnormalization6: BatchNorm<B>,
    conv1d22: Conv1d<B>,
    conv1d23: Conv1d<B>,
    averagepool1d6: AvgPool1d,
    conv1d24: Conv1d<B>,
    conv1d25: Conv1d<B>,
    batchnormalization7: BatchNorm<B>,
    conv1d26: Conv1d<B>,
    conv1d27: Conv1d<B>,
    averagepool1d7: AvgPool1d,
    conv1d28: Conv1d<B>,
    conv1d29: Conv1d<B>,
    batchnormalization8: BatchNorm<B>,
    conv1d30: Conv1d<B>,
    conv1d31: Conv1d<B>,
    averagepool1d8: AvgPool1d,
    conv1d32: Conv1d<B>,
    conv1d33: Conv1d<B>,
    batchnormalization9: BatchNorm<B>,
    conv1d34: Conv1d<B>,
    conv1d35: Conv1d<B>,
    averagepool1d9: AvgPool1d,
    conv1d36: Conv1d<B>,
    conv1d37: Conv1d<B>,
    batchnormalization10: BatchNorm<B>,
    conv1d38: Conv1d<B>,
    conv1d39: Conv1d<B>,
    averagepool1d10: AvgPool1d,
    conv1d40: Conv1d<B>,
    conv1d41: Conv1d<B>,
    batchnormalization11: BatchNorm<B>,
    conv1d42: Conv1d<B>,
    conv1d43: Conv1d<B>,
    averagepool1d11: AvgPool1d,
    conv1d44: Conv1d<B>,
    conv1d45: Conv1d<B>,
    batchnormalization12: BatchNorm<B>,
    conv1d46: Conv1d<B>,
    conv1d47: Conv1d<B>,
    averagepool1d12: AvgPool1d,
    conv1d48: Conv1d<B>,
    conv1d49: Conv1d<B>,
    batchnormalization13: BatchNorm<B>,
    conv1d50: Conv1d<B>,
    batchnormalization14: BatchNorm<B>,
    conv1d51: Conv1d<B>,
    conv1d52: Conv1d<B>,
    averagepool1d13: AvgPool1d,
    conv1d53: Conv1d<B>,
    conv1d54: Conv1d<B>,
    batchnormalization15: BatchNorm<B>,
    conv1d55: Conv1d<B>,
    conv1d56: Conv1d<B>,
    averagepool1d14: AvgPool1d,
    conv1d57: Conv1d<B>,
    conv1d58: Conv1d<B>,
    batchnormalization16: BatchNorm<B>,
    conv1d59: Conv1d<B>,
    conv1d60: Conv1d<B>,
    averagepool1d15: AvgPool1d,
    conv1d61: Conv1d<B>,
    conv1d62: Conv1d<B>,
    batchnormalization17: BatchNorm<B>,
    conv1d63: Conv1d<B>,
    conv1d64: Conv1d<B>,
    averagepool1d16: AvgPool1d,
    conv1d65: Conv1d<B>,
    conv1d66: Conv1d<B>,
    batchnormalization18: BatchNorm<B>,
    conv1d67: Conv1d<B>,
    conv1d68: Conv1d<B>,
    averagepool1d17: AvgPool1d,
    conv1d69: Conv1d<B>,
    conv1d70: Conv1d<B>,
    batchnormalization19: BatchNorm<B>,
    conv1d71: Conv1d<B>,
    conv1d72: Conv1d<B>,
    averagepool1d18: AvgPool1d,
    conv1d73: Conv1d<B>,
    conv1d74: Conv1d<B>,
    batchnormalization20: BatchNorm<B>,
    conv1d75: Conv1d<B>,
    conv1d76: Conv1d<B>,
    averagepool1d19: AvgPool1d,
    conv1d77: Conv1d<B>,
    conv1d78: Conv1d<B>,
    batchnormalization21: BatchNorm<B>,
    conv1d79: Conv1d<B>,
    conv1d80: Conv1d<B>,
    averagepool1d20: AvgPool1d,
    conv1d81: Conv1d<B>,
    conv1d82: Conv1d<B>,
    batchnormalization22: BatchNorm<B>,
    conv1d83: Conv1d<B>,
    conv1d84: Conv1d<B>,
    averagepool1d21: AvgPool1d,
    conv1d85: Conv1d<B>,
    conv1d86: Conv1d<B>,
    batchnormalization23: BatchNorm<B>,
    conv1d87: Conv1d<B>,
    conv1d88: Conv1d<B>,
    averagepool1d22: AvgPool1d,
    conv1d89: Conv1d<B>,
    conv1d90: Conv1d<B>,
    batchnormalization24: BatchNorm<B>,
    conv1d91: Conv1d<B>,
    conv1d92: Conv1d<B>,
    averagepool1d23: AvgPool1d,
    conv1d93: Conv1d<B>,
    conv1d94: Conv1d<B>,
    batchnormalization25: BatchNorm<B>,
    conv1d95: Conv1d<B>,
    conv1d96: Conv1d<B>,
    averagepool1d24: AvgPool1d,
    conv1d97: Conv1d<B>,
    conv1d98: Conv1d<B>,
    batchnormalization26: BatchNorm<B>,
    conv1d99: Conv1d<B>,
    conv1d100: Conv1d<B>,
    averagepool1d25: AvgPool1d,
    conv1d101: Conv1d<B>,
    conv1d102: Conv1d<B>,
    batchnormalization27: BatchNorm<B>,
    conv1d103: Conv1d<B>,
    conv1d104: Conv1d<B>,
    averagepool1d26: AvgPool1d,
    conv1d105: Conv1d<B>,
    conv1d106: Conv1d<B>,
    batchnormalization28: BatchNorm<B>,
    conv1d107: Conv1d<B>,
    conv1d108: Conv1d<B>,
    averagepool1d27: AvgPool1d,
    conv1d109: Conv1d<B>,
    conv1d110: Conv1d<B>,
    batchnormalization29: BatchNorm<B>,
    conv1d111: Conv1d<B>,
    conv1d112: Conv1d<B>,
    averagepool1d28: AvgPool1d,
    conv1d113: Conv1d<B>,
    conv1d114: Conv1d<B>,
    batchnormalization30: BatchNorm<B>,
    conv1d115: Conv1d<B>,
    conv1d116: Conv1d<B>,
    averagepool1d29: AvgPool1d,
    conv1d117: Conv1d<B>,
    conv1d118: Conv1d<B>,
    batchnormalization31: BatchNorm<B>,
    conv1d119: Conv1d<B>,
    conv1d120: Conv1d<B>,
    averagepool1d30: AvgPool1d,
    conv1d121: Conv1d<B>,
    conv1d122: Conv1d<B>,
    batchnormalization32: BatchNorm<B>,
    conv1d123: Conv1d<B>,
    conv1d124: Conv1d<B>,
    averagepool1d31: AvgPool1d,
    conv1d125: Conv1d<B>,
    conv1d126: Conv1d<B>,
    batchnormalization33: BatchNorm<B>,
    conv1d127: Conv1d<B>,
    conv1d128: Conv1d<B>,
    averagepool1d32: AvgPool1d,
    conv1d129: Conv1d<B>,
    conv1d130: Conv1d<B>,
    batchnormalization34: BatchNorm<B>,
    conv1d131: Conv1d<B>,
    conv1d132: Conv1d<B>,
    averagepool1d33: AvgPool1d,
    conv1d133: Conv1d<B>,
    conv1d134: Conv1d<B>,
    batchnormalization35: BatchNorm<B>,
    conv1d135: Conv1d<B>,
    conv1d136: Conv1d<B>,
    averagepool1d34: AvgPool1d,
    conv1d137: Conv1d<B>,
    conv1d138: Conv1d<B>,
    batchnormalization36: BatchNorm<B>,
    conv1d139: Conv1d<B>,
    conv1d140: Conv1d<B>,
    averagepool1d35: AvgPool1d,
    conv1d141: Conv1d<B>,
    conv1d142: Conv1d<B>,
    batchnormalization37: BatchNorm<B>,
    conv1d143: Conv1d<B>,
    conv1d144: Conv1d<B>,
    averagepool1d36: AvgPool1d,
    conv1d145: Conv1d<B>,
    conv1d146: Conv1d<B>,
    batchnormalization38: BatchNorm<B>,
    conv1d147: Conv1d<B>,
    batchnormalization39: BatchNorm<B>,
    conv1d148: Conv1d<B>,
    conv1d149: Conv1d<B>,
    averagepool1d37: AvgPool1d,
    conv1d150: Conv1d<B>,
    conv1d151: Conv1d<B>,
    batchnormalization40: BatchNorm<B>,
    conv1d152: Conv1d<B>,
    conv1d153: Conv1d<B>,
    averagepool1d38: AvgPool1d,
    conv1d154: Conv1d<B>,
    conv1d155: Conv1d<B>,
    batchnormalization41: BatchNorm<B>,
    conv1d156: Conv1d<B>,
    conv1d157: Conv1d<B>,
    averagepool1d39: AvgPool1d,
    conv1d158: Conv1d<B>,
    conv1d159: Conv1d<B>,
    batchnormalization42: BatchNorm<B>,
    conv1d160: Conv1d<B>,
    conv1d161: Conv1d<B>,
    averagepool1d40: AvgPool1d,
    conv1d162: Conv1d<B>,
    conv1d163: Conv1d<B>,
    batchnormalization43: BatchNorm<B>,
    conv1d164: Conv1d<B>,
    conv1d165: Conv1d<B>,
    averagepool1d41: AvgPool1d,
    conv1d166: Conv1d<B>,
    conv1d167: Conv1d<B>,
    batchnormalization44: BatchNorm<B>,
    conv1d168: Conv1d<B>,
    conv1d169: Conv1d<B>,
    averagepool1d42: AvgPool1d,
    conv1d170: Conv1d<B>,
    conv1d171: Conv1d<B>,
    batchnormalization45: BatchNorm<B>,
    conv1d172: Conv1d<B>,
    conv1d173: Conv1d<B>,
    averagepool1d43: AvgPool1d,
    conv1d174: Conv1d<B>,
    conv1d175: Conv1d<B>,
    batchnormalization46: BatchNorm<B>,
    conv1d176: Conv1d<B>,
    conv1d177: Conv1d<B>,
    averagepool1d44: AvgPool1d,
    conv1d178: Conv1d<B>,
    conv1d179: Conv1d<B>,
    batchnormalization47: BatchNorm<B>,
    conv1d180: Conv1d<B>,
    conv1d181: Conv1d<B>,
    averagepool1d45: AvgPool1d,
    conv1d182: Conv1d<B>,
    conv1d183: Conv1d<B>,
    batchnormalization48: BatchNorm<B>,
    conv1d184: Conv1d<B>,
    conv1d185: Conv1d<B>,
    averagepool1d46: AvgPool1d,
    conv1d186: Conv1d<B>,
    conv1d187: Conv1d<B>,
    batchnormalization49: BatchNorm<B>,
    conv1d188: Conv1d<B>,
    conv1d189: Conv1d<B>,
    averagepool1d47: AvgPool1d,
    conv1d190: Conv1d<B>,
    conv1d191: Conv1d<B>,
    batchnormalization50: BatchNorm<B>,
    conv1d192: Conv1d<B>,
    conv1d193: Conv1d<B>,
    averagepool1d48: AvgPool1d,
    conv1d194: Conv1d<B>,
    conv1d195: Conv1d<B>,
    batchnormalization51: BatchNorm<B>,
    conv1d196: Conv1d<B>,
    conv1d197: Conv1d<B>,
    averagepool1d49: AvgPool1d,
    conv1d198: Conv1d<B>,
    conv1d199: Conv1d<B>,
    batchnormalization52: BatchNorm<B>,
    conv1d200: Conv1d<B>,
    conv1d201: Conv1d<B>,
    averagepool1d50: AvgPool1d,
    conv1d202: Conv1d<B>,
    conv1d203: Conv1d<B>,
    batchnormalization53: BatchNorm<B>,
    conv1d204: Conv1d<B>,
    conv1d205: Conv1d<B>,
    averagepool1d51: AvgPool1d,
    conv1d206: Conv1d<B>,
    conv1d207: Conv1d<B>,
    batchnormalization54: BatchNorm<B>,
    conv1d208: Conv1d<B>,
    conv1d209: Conv1d<B>,
    averagepool1d52: AvgPool1d,
    conv1d210: Conv1d<B>,
    conv1d211: Conv1d<B>,
    batchnormalization55: BatchNorm<B>,
    conv1d212: Conv1d<B>,
    conv1d213: Conv1d<B>,
    batchnormalization56: BatchNorm<B>,
    phantom: core::marker::PhantomData<B>,
    device: burn::module::Ignored<B::Device>,
}


impl<B: Backend> Default for Model<B> {
    fn default() -> Self {
        Self::from_file("nn/speaker_identification/model.bpk", &Default::default())
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
        let conv2d1 = Conv2dConfig::new([1, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d2 = Conv2dConfig::new([32, 32], [3, 3])
            .with_stride([2, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d3 = Conv2dConfig::new([32, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d4 = Conv2dConfig::new([32, 32], [1, 1])
            .with_stride([2, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d5 = Conv2dConfig::new([32, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d6 = Conv2dConfig::new([32, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d7 = Conv2dConfig::new([32, 32], [3, 3])
            .with_stride([2, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d8 = Conv2dConfig::new([32, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d9 = Conv2dConfig::new([32, 32], [1, 1])
            .with_stride([2, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d10 = Conv2dConfig::new([32, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d11 = Conv2dConfig::new([32, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d12 = Conv2dConfig::new([32, 32], [3, 3])
            .with_stride([2, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d1 = Conv1dConfig::new(320, 128, 5)
            .with_stride(2)
            .with_padding(PaddingConfig1d::Explicit(2))
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization1 = BatchNormConfig::new(128)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d2 = Conv1dConfig::new(128, 128, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d3 = Conv1dConfig::new(128, 32, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(1))
            .with_dilation(1)
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let averagepool1d1 = AvgPool1dConfig::new(100)
            .with_stride(100)
            .with_padding(PaddingConfig1d::Valid)
            .with_count_include_pad(false)
            .with_ceil_mode(false)
            .init();
        let conv1d4 = Conv1dConfig::new(128, 64, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d5 = Conv1dConfig::new(64, 32, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization2 = BatchNormConfig::new(160)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d6 = Conv1dConfig::new(160, 128, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d7 = Conv1dConfig::new(128, 32, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(1))
            .with_dilation(1)
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let averagepool1d2 = AvgPool1dConfig::new(100)
            .with_stride(100)
            .with_padding(PaddingConfig1d::Valid)
            .with_count_include_pad(false)
            .with_ceil_mode(false)
            .init();
        let conv1d8 = Conv1dConfig::new(128, 64, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d9 = Conv1dConfig::new(64, 32, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization3 = BatchNormConfig::new(192)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d10 = Conv1dConfig::new(192, 128, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d11 = Conv1dConfig::new(128, 32, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(1))
            .with_dilation(1)
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let averagepool1d3 = AvgPool1dConfig::new(100)
            .with_stride(100)
            .with_padding(PaddingConfig1d::Valid)
            .with_count_include_pad(false)
            .with_ceil_mode(false)
            .init();
        let conv1d12 = Conv1dConfig::new(128, 64, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d13 = Conv1dConfig::new(64, 32, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization4 = BatchNormConfig::new(224)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d14 = Conv1dConfig::new(224, 128, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d15 = Conv1dConfig::new(128, 32, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(1))
            .with_dilation(1)
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let averagepool1d4 = AvgPool1dConfig::new(100)
            .with_stride(100)
            .with_padding(PaddingConfig1d::Valid)
            .with_count_include_pad(false)
            .with_ceil_mode(false)
            .init();
        let conv1d16 = Conv1dConfig::new(128, 64, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d17 = Conv1dConfig::new(64, 32, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization5 = BatchNormConfig::new(256)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d18 = Conv1dConfig::new(256, 128, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d19 = Conv1dConfig::new(128, 32, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(1))
            .with_dilation(1)
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let averagepool1d5 = AvgPool1dConfig::new(100)
            .with_stride(100)
            .with_padding(PaddingConfig1d::Valid)
            .with_count_include_pad(false)
            .with_ceil_mode(false)
            .init();
        let conv1d20 = Conv1dConfig::new(128, 64, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d21 = Conv1dConfig::new(64, 32, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization6 = BatchNormConfig::new(288)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d22 = Conv1dConfig::new(288, 128, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d23 = Conv1dConfig::new(128, 32, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(1))
            .with_dilation(1)
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let averagepool1d6 = AvgPool1dConfig::new(100)
            .with_stride(100)
            .with_padding(PaddingConfig1d::Valid)
            .with_count_include_pad(false)
            .with_ceil_mode(false)
            .init();
        let conv1d24 = Conv1dConfig::new(128, 64, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d25 = Conv1dConfig::new(64, 32, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization7 = BatchNormConfig::new(320)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d26 = Conv1dConfig::new(320, 128, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d27 = Conv1dConfig::new(128, 32, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(1))
            .with_dilation(1)
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let averagepool1d7 = AvgPool1dConfig::new(100)
            .with_stride(100)
            .with_padding(PaddingConfig1d::Valid)
            .with_count_include_pad(false)
            .with_ceil_mode(false)
            .init();
        let conv1d28 = Conv1dConfig::new(128, 64, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d29 = Conv1dConfig::new(64, 32, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization8 = BatchNormConfig::new(352)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d30 = Conv1dConfig::new(352, 128, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d31 = Conv1dConfig::new(128, 32, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(1))
            .with_dilation(1)
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let averagepool1d8 = AvgPool1dConfig::new(100)
            .with_stride(100)
            .with_padding(PaddingConfig1d::Valid)
            .with_count_include_pad(false)
            .with_ceil_mode(false)
            .init();
        let conv1d32 = Conv1dConfig::new(128, 64, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d33 = Conv1dConfig::new(64, 32, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization9 = BatchNormConfig::new(384)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d34 = Conv1dConfig::new(384, 128, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d35 = Conv1dConfig::new(128, 32, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(1))
            .with_dilation(1)
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let averagepool1d9 = AvgPool1dConfig::new(100)
            .with_stride(100)
            .with_padding(PaddingConfig1d::Valid)
            .with_count_include_pad(false)
            .with_ceil_mode(false)
            .init();
        let conv1d36 = Conv1dConfig::new(128, 64, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d37 = Conv1dConfig::new(64, 32, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization10 = BatchNormConfig::new(416)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d38 = Conv1dConfig::new(416, 128, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d39 = Conv1dConfig::new(128, 32, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(1))
            .with_dilation(1)
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let averagepool1d10 = AvgPool1dConfig::new(100)
            .with_stride(100)
            .with_padding(PaddingConfig1d::Valid)
            .with_count_include_pad(false)
            .with_ceil_mode(false)
            .init();
        let conv1d40 = Conv1dConfig::new(128, 64, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d41 = Conv1dConfig::new(64, 32, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization11 = BatchNormConfig::new(448)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d42 = Conv1dConfig::new(448, 128, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d43 = Conv1dConfig::new(128, 32, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(1))
            .with_dilation(1)
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let averagepool1d11 = AvgPool1dConfig::new(100)
            .with_stride(100)
            .with_padding(PaddingConfig1d::Valid)
            .with_count_include_pad(false)
            .with_ceil_mode(false)
            .init();
        let conv1d44 = Conv1dConfig::new(128, 64, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d45 = Conv1dConfig::new(64, 32, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization12 = BatchNormConfig::new(480)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d46 = Conv1dConfig::new(480, 128, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d47 = Conv1dConfig::new(128, 32, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(1))
            .with_dilation(1)
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let averagepool1d12 = AvgPool1dConfig::new(100)
            .with_stride(100)
            .with_padding(PaddingConfig1d::Valid)
            .with_count_include_pad(false)
            .with_ceil_mode(false)
            .init();
        let conv1d48 = Conv1dConfig::new(128, 64, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d49 = Conv1dConfig::new(64, 32, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization13 = BatchNormConfig::new(512)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d50 = Conv1dConfig::new(512, 256, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization14 = BatchNormConfig::new(256)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d51 = Conv1dConfig::new(256, 128, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d52 = Conv1dConfig::new(128, 32, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(2))
            .with_dilation(2)
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let averagepool1d13 = AvgPool1dConfig::new(100)
            .with_stride(100)
            .with_padding(PaddingConfig1d::Valid)
            .with_count_include_pad(false)
            .with_ceil_mode(false)
            .init();
        let conv1d53 = Conv1dConfig::new(128, 64, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d54 = Conv1dConfig::new(64, 32, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization15 = BatchNormConfig::new(288)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d55 = Conv1dConfig::new(288, 128, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d56 = Conv1dConfig::new(128, 32, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(2))
            .with_dilation(2)
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let averagepool1d14 = AvgPool1dConfig::new(100)
            .with_stride(100)
            .with_padding(PaddingConfig1d::Valid)
            .with_count_include_pad(false)
            .with_ceil_mode(false)
            .init();
        let conv1d57 = Conv1dConfig::new(128, 64, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d58 = Conv1dConfig::new(64, 32, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization16 = BatchNormConfig::new(320)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d59 = Conv1dConfig::new(320, 128, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d60 = Conv1dConfig::new(128, 32, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(2))
            .with_dilation(2)
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let averagepool1d15 = AvgPool1dConfig::new(100)
            .with_stride(100)
            .with_padding(PaddingConfig1d::Valid)
            .with_count_include_pad(false)
            .with_ceil_mode(false)
            .init();
        let conv1d61 = Conv1dConfig::new(128, 64, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d62 = Conv1dConfig::new(64, 32, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization17 = BatchNormConfig::new(352)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d63 = Conv1dConfig::new(352, 128, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d64 = Conv1dConfig::new(128, 32, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(2))
            .with_dilation(2)
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let averagepool1d16 = AvgPool1dConfig::new(100)
            .with_stride(100)
            .with_padding(PaddingConfig1d::Valid)
            .with_count_include_pad(false)
            .with_ceil_mode(false)
            .init();
        let conv1d65 = Conv1dConfig::new(128, 64, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d66 = Conv1dConfig::new(64, 32, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization18 = BatchNormConfig::new(384)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d67 = Conv1dConfig::new(384, 128, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d68 = Conv1dConfig::new(128, 32, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(2))
            .with_dilation(2)
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let averagepool1d17 = AvgPool1dConfig::new(100)
            .with_stride(100)
            .with_padding(PaddingConfig1d::Valid)
            .with_count_include_pad(false)
            .with_ceil_mode(false)
            .init();
        let conv1d69 = Conv1dConfig::new(128, 64, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d70 = Conv1dConfig::new(64, 32, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization19 = BatchNormConfig::new(416)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d71 = Conv1dConfig::new(416, 128, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d72 = Conv1dConfig::new(128, 32, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(2))
            .with_dilation(2)
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let averagepool1d18 = AvgPool1dConfig::new(100)
            .with_stride(100)
            .with_padding(PaddingConfig1d::Valid)
            .with_count_include_pad(false)
            .with_ceil_mode(false)
            .init();
        let conv1d73 = Conv1dConfig::new(128, 64, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d74 = Conv1dConfig::new(64, 32, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization20 = BatchNormConfig::new(448)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d75 = Conv1dConfig::new(448, 128, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d76 = Conv1dConfig::new(128, 32, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(2))
            .with_dilation(2)
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let averagepool1d19 = AvgPool1dConfig::new(100)
            .with_stride(100)
            .with_padding(PaddingConfig1d::Valid)
            .with_count_include_pad(false)
            .with_ceil_mode(false)
            .init();
        let conv1d77 = Conv1dConfig::new(128, 64, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d78 = Conv1dConfig::new(64, 32, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization21 = BatchNormConfig::new(480)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d79 = Conv1dConfig::new(480, 128, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d80 = Conv1dConfig::new(128, 32, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(2))
            .with_dilation(2)
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let averagepool1d20 = AvgPool1dConfig::new(100)
            .with_stride(100)
            .with_padding(PaddingConfig1d::Valid)
            .with_count_include_pad(false)
            .with_ceil_mode(false)
            .init();
        let conv1d81 = Conv1dConfig::new(128, 64, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d82 = Conv1dConfig::new(64, 32, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization22 = BatchNormConfig::new(512)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d83 = Conv1dConfig::new(512, 128, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d84 = Conv1dConfig::new(128, 32, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(2))
            .with_dilation(2)
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let averagepool1d21 = AvgPool1dConfig::new(100)
            .with_stride(100)
            .with_padding(PaddingConfig1d::Valid)
            .with_count_include_pad(false)
            .with_ceil_mode(false)
            .init();
        let conv1d85 = Conv1dConfig::new(128, 64, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d86 = Conv1dConfig::new(64, 32, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization23 = BatchNormConfig::new(544)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d87 = Conv1dConfig::new(544, 128, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d88 = Conv1dConfig::new(128, 32, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(2))
            .with_dilation(2)
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let averagepool1d22 = AvgPool1dConfig::new(100)
            .with_stride(100)
            .with_padding(PaddingConfig1d::Valid)
            .with_count_include_pad(false)
            .with_ceil_mode(false)
            .init();
        let conv1d89 = Conv1dConfig::new(128, 64, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d90 = Conv1dConfig::new(64, 32, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization24 = BatchNormConfig::new(576)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d91 = Conv1dConfig::new(576, 128, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d92 = Conv1dConfig::new(128, 32, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(2))
            .with_dilation(2)
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let averagepool1d23 = AvgPool1dConfig::new(100)
            .with_stride(100)
            .with_padding(PaddingConfig1d::Valid)
            .with_count_include_pad(false)
            .with_ceil_mode(false)
            .init();
        let conv1d93 = Conv1dConfig::new(128, 64, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d94 = Conv1dConfig::new(64, 32, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization25 = BatchNormConfig::new(608)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d95 = Conv1dConfig::new(608, 128, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d96 = Conv1dConfig::new(128, 32, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(2))
            .with_dilation(2)
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let averagepool1d24 = AvgPool1dConfig::new(100)
            .with_stride(100)
            .with_padding(PaddingConfig1d::Valid)
            .with_count_include_pad(false)
            .with_ceil_mode(false)
            .init();
        let conv1d97 = Conv1dConfig::new(128, 64, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d98 = Conv1dConfig::new(64, 32, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization26 = BatchNormConfig::new(640)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d99 = Conv1dConfig::new(640, 128, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d100 = Conv1dConfig::new(128, 32, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(2))
            .with_dilation(2)
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let averagepool1d25 = AvgPool1dConfig::new(100)
            .with_stride(100)
            .with_padding(PaddingConfig1d::Valid)
            .with_count_include_pad(false)
            .with_ceil_mode(false)
            .init();
        let conv1d101 = Conv1dConfig::new(128, 64, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d102 = Conv1dConfig::new(64, 32, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization27 = BatchNormConfig::new(672)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d103 = Conv1dConfig::new(672, 128, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d104 = Conv1dConfig::new(128, 32, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(2))
            .with_dilation(2)
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let averagepool1d26 = AvgPool1dConfig::new(100)
            .with_stride(100)
            .with_padding(PaddingConfig1d::Valid)
            .with_count_include_pad(false)
            .with_ceil_mode(false)
            .init();
        let conv1d105 = Conv1dConfig::new(128, 64, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d106 = Conv1dConfig::new(64, 32, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization28 = BatchNormConfig::new(704)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d107 = Conv1dConfig::new(704, 128, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d108 = Conv1dConfig::new(128, 32, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(2))
            .with_dilation(2)
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let averagepool1d27 = AvgPool1dConfig::new(100)
            .with_stride(100)
            .with_padding(PaddingConfig1d::Valid)
            .with_count_include_pad(false)
            .with_ceil_mode(false)
            .init();
        let conv1d109 = Conv1dConfig::new(128, 64, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d110 = Conv1dConfig::new(64, 32, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization29 = BatchNormConfig::new(736)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d111 = Conv1dConfig::new(736, 128, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d112 = Conv1dConfig::new(128, 32, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(2))
            .with_dilation(2)
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let averagepool1d28 = AvgPool1dConfig::new(100)
            .with_stride(100)
            .with_padding(PaddingConfig1d::Valid)
            .with_count_include_pad(false)
            .with_ceil_mode(false)
            .init();
        let conv1d113 = Conv1dConfig::new(128, 64, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d114 = Conv1dConfig::new(64, 32, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization30 = BatchNormConfig::new(768)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d115 = Conv1dConfig::new(768, 128, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d116 = Conv1dConfig::new(128, 32, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(2))
            .with_dilation(2)
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let averagepool1d29 = AvgPool1dConfig::new(100)
            .with_stride(100)
            .with_padding(PaddingConfig1d::Valid)
            .with_count_include_pad(false)
            .with_ceil_mode(false)
            .init();
        let conv1d117 = Conv1dConfig::new(128, 64, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d118 = Conv1dConfig::new(64, 32, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization31 = BatchNormConfig::new(800)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d119 = Conv1dConfig::new(800, 128, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d120 = Conv1dConfig::new(128, 32, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(2))
            .with_dilation(2)
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let averagepool1d30 = AvgPool1dConfig::new(100)
            .with_stride(100)
            .with_padding(PaddingConfig1d::Valid)
            .with_count_include_pad(false)
            .with_ceil_mode(false)
            .init();
        let conv1d121 = Conv1dConfig::new(128, 64, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d122 = Conv1dConfig::new(64, 32, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization32 = BatchNormConfig::new(832)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d123 = Conv1dConfig::new(832, 128, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d124 = Conv1dConfig::new(128, 32, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(2))
            .with_dilation(2)
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let averagepool1d31 = AvgPool1dConfig::new(100)
            .with_stride(100)
            .with_padding(PaddingConfig1d::Valid)
            .with_count_include_pad(false)
            .with_ceil_mode(false)
            .init();
        let conv1d125 = Conv1dConfig::new(128, 64, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d126 = Conv1dConfig::new(64, 32, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization33 = BatchNormConfig::new(864)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d127 = Conv1dConfig::new(864, 128, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d128 = Conv1dConfig::new(128, 32, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(2))
            .with_dilation(2)
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let averagepool1d32 = AvgPool1dConfig::new(100)
            .with_stride(100)
            .with_padding(PaddingConfig1d::Valid)
            .with_count_include_pad(false)
            .with_ceil_mode(false)
            .init();
        let conv1d129 = Conv1dConfig::new(128, 64, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d130 = Conv1dConfig::new(64, 32, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization34 = BatchNormConfig::new(896)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d131 = Conv1dConfig::new(896, 128, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d132 = Conv1dConfig::new(128, 32, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(2))
            .with_dilation(2)
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let averagepool1d33 = AvgPool1dConfig::new(100)
            .with_stride(100)
            .with_padding(PaddingConfig1d::Valid)
            .with_count_include_pad(false)
            .with_ceil_mode(false)
            .init();
        let conv1d133 = Conv1dConfig::new(128, 64, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d134 = Conv1dConfig::new(64, 32, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization35 = BatchNormConfig::new(928)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d135 = Conv1dConfig::new(928, 128, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d136 = Conv1dConfig::new(128, 32, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(2))
            .with_dilation(2)
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let averagepool1d34 = AvgPool1dConfig::new(100)
            .with_stride(100)
            .with_padding(PaddingConfig1d::Valid)
            .with_count_include_pad(false)
            .with_ceil_mode(false)
            .init();
        let conv1d137 = Conv1dConfig::new(128, 64, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d138 = Conv1dConfig::new(64, 32, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization36 = BatchNormConfig::new(960)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d139 = Conv1dConfig::new(960, 128, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d140 = Conv1dConfig::new(128, 32, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(2))
            .with_dilation(2)
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let averagepool1d35 = AvgPool1dConfig::new(100)
            .with_stride(100)
            .with_padding(PaddingConfig1d::Valid)
            .with_count_include_pad(false)
            .with_ceil_mode(false)
            .init();
        let conv1d141 = Conv1dConfig::new(128, 64, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d142 = Conv1dConfig::new(64, 32, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization37 = BatchNormConfig::new(992)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d143 = Conv1dConfig::new(992, 128, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d144 = Conv1dConfig::new(128, 32, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(2))
            .with_dilation(2)
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let averagepool1d36 = AvgPool1dConfig::new(100)
            .with_stride(100)
            .with_padding(PaddingConfig1d::Valid)
            .with_count_include_pad(false)
            .with_ceil_mode(false)
            .init();
        let conv1d145 = Conv1dConfig::new(128, 64, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d146 = Conv1dConfig::new(64, 32, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization38 = BatchNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d147 = Conv1dConfig::new(1024, 512, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization39 = BatchNormConfig::new(512)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d148 = Conv1dConfig::new(512, 128, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d149 = Conv1dConfig::new(128, 32, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(2))
            .with_dilation(2)
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let averagepool1d37 = AvgPool1dConfig::new(100)
            .with_stride(100)
            .with_padding(PaddingConfig1d::Valid)
            .with_count_include_pad(false)
            .with_ceil_mode(false)
            .init();
        let conv1d150 = Conv1dConfig::new(128, 64, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d151 = Conv1dConfig::new(64, 32, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization40 = BatchNormConfig::new(544)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d152 = Conv1dConfig::new(544, 128, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d153 = Conv1dConfig::new(128, 32, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(2))
            .with_dilation(2)
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let averagepool1d38 = AvgPool1dConfig::new(100)
            .with_stride(100)
            .with_padding(PaddingConfig1d::Valid)
            .with_count_include_pad(false)
            .with_ceil_mode(false)
            .init();
        let conv1d154 = Conv1dConfig::new(128, 64, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d155 = Conv1dConfig::new(64, 32, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization41 = BatchNormConfig::new(576)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d156 = Conv1dConfig::new(576, 128, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d157 = Conv1dConfig::new(128, 32, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(2))
            .with_dilation(2)
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let averagepool1d39 = AvgPool1dConfig::new(100)
            .with_stride(100)
            .with_padding(PaddingConfig1d::Valid)
            .with_count_include_pad(false)
            .with_ceil_mode(false)
            .init();
        let conv1d158 = Conv1dConfig::new(128, 64, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d159 = Conv1dConfig::new(64, 32, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization42 = BatchNormConfig::new(608)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d160 = Conv1dConfig::new(608, 128, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d161 = Conv1dConfig::new(128, 32, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(2))
            .with_dilation(2)
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let averagepool1d40 = AvgPool1dConfig::new(100)
            .with_stride(100)
            .with_padding(PaddingConfig1d::Valid)
            .with_count_include_pad(false)
            .with_ceil_mode(false)
            .init();
        let conv1d162 = Conv1dConfig::new(128, 64, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d163 = Conv1dConfig::new(64, 32, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization43 = BatchNormConfig::new(640)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d164 = Conv1dConfig::new(640, 128, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d165 = Conv1dConfig::new(128, 32, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(2))
            .with_dilation(2)
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let averagepool1d41 = AvgPool1dConfig::new(100)
            .with_stride(100)
            .with_padding(PaddingConfig1d::Valid)
            .with_count_include_pad(false)
            .with_ceil_mode(false)
            .init();
        let conv1d166 = Conv1dConfig::new(128, 64, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d167 = Conv1dConfig::new(64, 32, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization44 = BatchNormConfig::new(672)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d168 = Conv1dConfig::new(672, 128, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d169 = Conv1dConfig::new(128, 32, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(2))
            .with_dilation(2)
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let averagepool1d42 = AvgPool1dConfig::new(100)
            .with_stride(100)
            .with_padding(PaddingConfig1d::Valid)
            .with_count_include_pad(false)
            .with_ceil_mode(false)
            .init();
        let conv1d170 = Conv1dConfig::new(128, 64, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d171 = Conv1dConfig::new(64, 32, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization45 = BatchNormConfig::new(704)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d172 = Conv1dConfig::new(704, 128, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d173 = Conv1dConfig::new(128, 32, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(2))
            .with_dilation(2)
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let averagepool1d43 = AvgPool1dConfig::new(100)
            .with_stride(100)
            .with_padding(PaddingConfig1d::Valid)
            .with_count_include_pad(false)
            .with_ceil_mode(false)
            .init();
        let conv1d174 = Conv1dConfig::new(128, 64, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d175 = Conv1dConfig::new(64, 32, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization46 = BatchNormConfig::new(736)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d176 = Conv1dConfig::new(736, 128, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d177 = Conv1dConfig::new(128, 32, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(2))
            .with_dilation(2)
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let averagepool1d44 = AvgPool1dConfig::new(100)
            .with_stride(100)
            .with_padding(PaddingConfig1d::Valid)
            .with_count_include_pad(false)
            .with_ceil_mode(false)
            .init();
        let conv1d178 = Conv1dConfig::new(128, 64, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d179 = Conv1dConfig::new(64, 32, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization47 = BatchNormConfig::new(768)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d180 = Conv1dConfig::new(768, 128, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d181 = Conv1dConfig::new(128, 32, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(2))
            .with_dilation(2)
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let averagepool1d45 = AvgPool1dConfig::new(100)
            .with_stride(100)
            .with_padding(PaddingConfig1d::Valid)
            .with_count_include_pad(false)
            .with_ceil_mode(false)
            .init();
        let conv1d182 = Conv1dConfig::new(128, 64, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d183 = Conv1dConfig::new(64, 32, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization48 = BatchNormConfig::new(800)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d184 = Conv1dConfig::new(800, 128, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d185 = Conv1dConfig::new(128, 32, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(2))
            .with_dilation(2)
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let averagepool1d46 = AvgPool1dConfig::new(100)
            .with_stride(100)
            .with_padding(PaddingConfig1d::Valid)
            .with_count_include_pad(false)
            .with_ceil_mode(false)
            .init();
        let conv1d186 = Conv1dConfig::new(128, 64, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d187 = Conv1dConfig::new(64, 32, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization49 = BatchNormConfig::new(832)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d188 = Conv1dConfig::new(832, 128, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d189 = Conv1dConfig::new(128, 32, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(2))
            .with_dilation(2)
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let averagepool1d47 = AvgPool1dConfig::new(100)
            .with_stride(100)
            .with_padding(PaddingConfig1d::Valid)
            .with_count_include_pad(false)
            .with_ceil_mode(false)
            .init();
        let conv1d190 = Conv1dConfig::new(128, 64, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d191 = Conv1dConfig::new(64, 32, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization50 = BatchNormConfig::new(864)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d192 = Conv1dConfig::new(864, 128, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d193 = Conv1dConfig::new(128, 32, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(2))
            .with_dilation(2)
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let averagepool1d48 = AvgPool1dConfig::new(100)
            .with_stride(100)
            .with_padding(PaddingConfig1d::Valid)
            .with_count_include_pad(false)
            .with_ceil_mode(false)
            .init();
        let conv1d194 = Conv1dConfig::new(128, 64, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d195 = Conv1dConfig::new(64, 32, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization51 = BatchNormConfig::new(896)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d196 = Conv1dConfig::new(896, 128, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d197 = Conv1dConfig::new(128, 32, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(2))
            .with_dilation(2)
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let averagepool1d49 = AvgPool1dConfig::new(100)
            .with_stride(100)
            .with_padding(PaddingConfig1d::Valid)
            .with_count_include_pad(false)
            .with_ceil_mode(false)
            .init();
        let conv1d198 = Conv1dConfig::new(128, 64, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d199 = Conv1dConfig::new(64, 32, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization52 = BatchNormConfig::new(928)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d200 = Conv1dConfig::new(928, 128, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d201 = Conv1dConfig::new(128, 32, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(2))
            .with_dilation(2)
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let averagepool1d50 = AvgPool1dConfig::new(100)
            .with_stride(100)
            .with_padding(PaddingConfig1d::Valid)
            .with_count_include_pad(false)
            .with_ceil_mode(false)
            .init();
        let conv1d202 = Conv1dConfig::new(128, 64, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d203 = Conv1dConfig::new(64, 32, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization53 = BatchNormConfig::new(960)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d204 = Conv1dConfig::new(960, 128, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d205 = Conv1dConfig::new(128, 32, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(2))
            .with_dilation(2)
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let averagepool1d51 = AvgPool1dConfig::new(100)
            .with_stride(100)
            .with_padding(PaddingConfig1d::Valid)
            .with_count_include_pad(false)
            .with_ceil_mode(false)
            .init();
        let conv1d206 = Conv1dConfig::new(128, 64, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d207 = Conv1dConfig::new(64, 32, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization54 = BatchNormConfig::new(992)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d208 = Conv1dConfig::new(992, 128, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d209 = Conv1dConfig::new(128, 32, 3)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(2))
            .with_dilation(2)
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let averagepool1d52 = AvgPool1dConfig::new(100)
            .with_stride(100)
            .with_padding(PaddingConfig1d::Valid)
            .with_count_include_pad(false)
            .with_ceil_mode(false)
            .init();
        let conv1d210 = Conv1dConfig::new(128, 64, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d211 = Conv1dConfig::new(64, 32, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let batchnormalization55 = BatchNormConfig::new(1024)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv1d212 = Conv1dConfig::new(1024, 512, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv1d213 = Conv1dConfig::new(1024, 512, 1)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Valid)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization56 = BatchNormConfig::new(512)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        Self {
            conv2d1,
            conv2d2,
            conv2d3,
            conv2d4,
            conv2d5,
            conv2d6,
            conv2d7,
            conv2d8,
            conv2d9,
            conv2d10,
            conv2d11,
            conv2d12,
            conv1d1,
            batchnormalization1,
            conv1d2,
            conv1d3,
            averagepool1d1,
            conv1d4,
            conv1d5,
            batchnormalization2,
            conv1d6,
            conv1d7,
            averagepool1d2,
            conv1d8,
            conv1d9,
            batchnormalization3,
            conv1d10,
            conv1d11,
            averagepool1d3,
            conv1d12,
            conv1d13,
            batchnormalization4,
            conv1d14,
            conv1d15,
            averagepool1d4,
            conv1d16,
            conv1d17,
            batchnormalization5,
            conv1d18,
            conv1d19,
            averagepool1d5,
            conv1d20,
            conv1d21,
            batchnormalization6,
            conv1d22,
            conv1d23,
            averagepool1d6,
            conv1d24,
            conv1d25,
            batchnormalization7,
            conv1d26,
            conv1d27,
            averagepool1d7,
            conv1d28,
            conv1d29,
            batchnormalization8,
            conv1d30,
            conv1d31,
            averagepool1d8,
            conv1d32,
            conv1d33,
            batchnormalization9,
            conv1d34,
            conv1d35,
            averagepool1d9,
            conv1d36,
            conv1d37,
            batchnormalization10,
            conv1d38,
            conv1d39,
            averagepool1d10,
            conv1d40,
            conv1d41,
            batchnormalization11,
            conv1d42,
            conv1d43,
            averagepool1d11,
            conv1d44,
            conv1d45,
            batchnormalization12,
            conv1d46,
            conv1d47,
            averagepool1d12,
            conv1d48,
            conv1d49,
            batchnormalization13,
            conv1d50,
            batchnormalization14,
            conv1d51,
            conv1d52,
            averagepool1d13,
            conv1d53,
            conv1d54,
            batchnormalization15,
            conv1d55,
            conv1d56,
            averagepool1d14,
            conv1d57,
            conv1d58,
            batchnormalization16,
            conv1d59,
            conv1d60,
            averagepool1d15,
            conv1d61,
            conv1d62,
            batchnormalization17,
            conv1d63,
            conv1d64,
            averagepool1d16,
            conv1d65,
            conv1d66,
            batchnormalization18,
            conv1d67,
            conv1d68,
            averagepool1d17,
            conv1d69,
            conv1d70,
            batchnormalization19,
            conv1d71,
            conv1d72,
            averagepool1d18,
            conv1d73,
            conv1d74,
            batchnormalization20,
            conv1d75,
            conv1d76,
            averagepool1d19,
            conv1d77,
            conv1d78,
            batchnormalization21,
            conv1d79,
            conv1d80,
            averagepool1d20,
            conv1d81,
            conv1d82,
            batchnormalization22,
            conv1d83,
            conv1d84,
            averagepool1d21,
            conv1d85,
            conv1d86,
            batchnormalization23,
            conv1d87,
            conv1d88,
            averagepool1d22,
            conv1d89,
            conv1d90,
            batchnormalization24,
            conv1d91,
            conv1d92,
            averagepool1d23,
            conv1d93,
            conv1d94,
            batchnormalization25,
            conv1d95,
            conv1d96,
            averagepool1d24,
            conv1d97,
            conv1d98,
            batchnormalization26,
            conv1d99,
            conv1d100,
            averagepool1d25,
            conv1d101,
            conv1d102,
            batchnormalization27,
            conv1d103,
            conv1d104,
            averagepool1d26,
            conv1d105,
            conv1d106,
            batchnormalization28,
            conv1d107,
            conv1d108,
            averagepool1d27,
            conv1d109,
            conv1d110,
            batchnormalization29,
            conv1d111,
            conv1d112,
            averagepool1d28,
            conv1d113,
            conv1d114,
            batchnormalization30,
            conv1d115,
            conv1d116,
            averagepool1d29,
            conv1d117,
            conv1d118,
            batchnormalization31,
            conv1d119,
            conv1d120,
            averagepool1d30,
            conv1d121,
            conv1d122,
            batchnormalization32,
            conv1d123,
            conv1d124,
            averagepool1d31,
            conv1d125,
            conv1d126,
            batchnormalization33,
            conv1d127,
            conv1d128,
            averagepool1d32,
            conv1d129,
            conv1d130,
            batchnormalization34,
            conv1d131,
            conv1d132,
            averagepool1d33,
            conv1d133,
            conv1d134,
            batchnormalization35,
            conv1d135,
            conv1d136,
            averagepool1d34,
            conv1d137,
            conv1d138,
            batchnormalization36,
            conv1d139,
            conv1d140,
            averagepool1d35,
            conv1d141,
            conv1d142,
            batchnormalization37,
            conv1d143,
            conv1d144,
            averagepool1d36,
            conv1d145,
            conv1d146,
            batchnormalization38,
            conv1d147,
            batchnormalization39,
            conv1d148,
            conv1d149,
            averagepool1d37,
            conv1d150,
            conv1d151,
            batchnormalization40,
            conv1d152,
            conv1d153,
            averagepool1d38,
            conv1d154,
            conv1d155,
            batchnormalization41,
            conv1d156,
            conv1d157,
            averagepool1d39,
            conv1d158,
            conv1d159,
            batchnormalization42,
            conv1d160,
            conv1d161,
            averagepool1d40,
            conv1d162,
            conv1d163,
            batchnormalization43,
            conv1d164,
            conv1d165,
            averagepool1d41,
            conv1d166,
            conv1d167,
            batchnormalization44,
            conv1d168,
            conv1d169,
            averagepool1d42,
            conv1d170,
            conv1d171,
            batchnormalization45,
            conv1d172,
            conv1d173,
            averagepool1d43,
            conv1d174,
            conv1d175,
            batchnormalization46,
            conv1d176,
            conv1d177,
            averagepool1d44,
            conv1d178,
            conv1d179,
            batchnormalization47,
            conv1d180,
            conv1d181,
            averagepool1d45,
            conv1d182,
            conv1d183,
            batchnormalization48,
            conv1d184,
            conv1d185,
            averagepool1d46,
            conv1d186,
            conv1d187,
            batchnormalization49,
            conv1d188,
            conv1d189,
            averagepool1d47,
            conv1d190,
            conv1d191,
            batchnormalization50,
            conv1d192,
            conv1d193,
            averagepool1d48,
            conv1d194,
            conv1d195,
            batchnormalization51,
            conv1d196,
            conv1d197,
            averagepool1d49,
            conv1d198,
            conv1d199,
            batchnormalization52,
            conv1d200,
            conv1d201,
            averagepool1d50,
            conv1d202,
            conv1d203,
            batchnormalization53,
            conv1d204,
            conv1d205,
            averagepool1d51,
            conv1d206,
            conv1d207,
            batchnormalization54,
            conv1d208,
            conv1d209,
            averagepool1d52,
            conv1d210,
            conv1d211,
            batchnormalization55,
            conv1d212,
            conv1d213,
            batchnormalization56,
            phantom: core::marker::PhantomData,
            device: burn::module::Ignored(device.clone()),
        }
    }

    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, feats: Tensor<B, 3>) -> Tensor<B, 2> {
        let constant622_out1 = 0.0000001f32;
        let constant629_out1 = 100f32;
        let constant630_out1 = 99f32;
        let transpose1_out1 = feats.permute([0, 2, 1]);
        let unsqueeze1_out1: Tensor<B, 4> = transpose1_out1.unsqueeze_dims::<4>(&[1]);
        let conv2d1_out1 = self.conv2d1.forward(unsqueeze1_out1);
        let relu1_out1 = burn::tensor::activation::relu(conv2d1_out1);
        let conv2d2_out1 = self.conv2d2.forward(relu1_out1.clone());
        let relu2_out1 = burn::tensor::activation::relu(conv2d2_out1);
        let conv2d3_out1 = self.conv2d3.forward(relu2_out1);
        let conv2d4_out1 = self.conv2d4.forward(relu1_out1);
        let add1_out1 = conv2d3_out1.add(conv2d4_out1);
        let relu3_out1 = burn::tensor::activation::relu(add1_out1);
        let conv2d5_out1 = self.conv2d5.forward(relu3_out1.clone());
        let relu4_out1 = burn::tensor::activation::relu(conv2d5_out1);
        let conv2d6_out1 = self.conv2d6.forward(relu4_out1);
        let add2_out1 = conv2d6_out1.add(relu3_out1);
        let relu5_out1 = burn::tensor::activation::relu(add2_out1);
        let conv2d7_out1 = self.conv2d7.forward(relu5_out1.clone());
        let relu6_out1 = burn::tensor::activation::relu(conv2d7_out1);
        let conv2d8_out1 = self.conv2d8.forward(relu6_out1);
        let conv2d9_out1 = self.conv2d9.forward(relu5_out1);
        let add3_out1 = conv2d8_out1.add(conv2d9_out1);
        let relu7_out1 = burn::tensor::activation::relu(add3_out1);
        let conv2d10_out1 = self.conv2d10.forward(relu7_out1.clone());
        let relu8_out1 = burn::tensor::activation::relu(conv2d10_out1);
        let conv2d11_out1 = self.conv2d11.forward(relu8_out1);
        let add4_out1 = conv2d11_out1.add(relu7_out1);
        let relu9_out1 = burn::tensor::activation::relu(add4_out1);
        let conv2d12_out1 = self.conv2d12.forward(relu9_out1);
        let relu10_out1 = burn::tensor::activation::relu(conv2d12_out1);
        let reshape1_out1 = relu10_out1.reshape([1, 320, 200]);
        let conv1d1_out1 = self.conv1d1.forward(reshape1_out1);
        let relu11_out1 = burn::tensor::activation::relu(conv1d1_out1);
        let batchnormalization1_out1 = self
            .batchnormalization1
            .forward(relu11_out1.clone());
        let relu12_out1 = burn::tensor::activation::relu(batchnormalization1_out1);
        let conv1d2_out1 = self.conv1d2.forward(relu12_out1);
        let relu13_out1 = burn::tensor::activation::relu(conv1d2_out1);
        let conv1d3_out1 = self.conv1d3.forward(relu13_out1.clone());
        let reducemean1_out1 = { relu13_out1.clone().mean_dim(2usize) };
        let averagepool1d1_out1 = self.averagepool1d1.forward(relu13_out1);
        let unsqueeze2_out1: Tensor<B, 4> = averagepool1d1_out1
            .unsqueeze_dims::<4>(&[-1]);
        let expand1_out1 = {
            let onnx_shape: [i64; 4usize] = [1, 128, 1, 100];
            let input_dims = unsqueeze2_out1.dims();
            let mut shape = onnx_shape;
            #[allow(clippy::needless_range_loop)]
            for i in 0..4usize {
                let dim_offset = 4usize - 4usize + i;
                if shape[dim_offset] == 1 && input_dims[i] > 1 {
                    shape[dim_offset] = input_dims[i] as i64;
                }
            }
            unsqueeze2_out1.expand(shape)
        };
        let reshape2_out1 = expand1_out1.reshape([1, 128, -1]);
        let slice1_out1 = reshape2_out1.slice(s![.., .., 0..100]);
        let add5_out1 = reducemean1_out1.add(slice1_out1);
        let conv1d4_out1 = self.conv1d4.forward(add5_out1);
        let relu14_out1 = burn::tensor::activation::relu(conv1d4_out1);
        let conv1d5_out1 = self.conv1d5.forward(relu14_out1);
        let sigmoid1_out1 = burn::tensor::activation::sigmoid(conv1d5_out1);
        let mul1_out1 = conv1d3_out1.mul(sigmoid1_out1);
        let concat1_out1 = burn::tensor::Tensor::cat([relu11_out1, mul1_out1].into(), 1);
        let batchnormalization2_out1 = self
            .batchnormalization2
            .forward(concat1_out1.clone());
        let relu15_out1 = burn::tensor::activation::relu(batchnormalization2_out1);
        let conv1d6_out1 = self.conv1d6.forward(relu15_out1);
        let relu16_out1 = burn::tensor::activation::relu(conv1d6_out1);
        let conv1d7_out1 = self.conv1d7.forward(relu16_out1.clone());
        let reducemean2_out1 = { relu16_out1.clone().mean_dim(2usize) };
        let averagepool1d2_out1 = self.averagepool1d2.forward(relu16_out1);
        let unsqueeze3_out1: Tensor<B, 4> = averagepool1d2_out1
            .unsqueeze_dims::<4>(&[-1]);
        let expand2_out1 = {
            let onnx_shape: [i64; 4usize] = [1, 128, 1, 100];
            let input_dims = unsqueeze3_out1.dims();
            let mut shape = onnx_shape;
            #[allow(clippy::needless_range_loop)]
            for i in 0..4usize {
                let dim_offset = 4usize - 4usize + i;
                if shape[dim_offset] == 1 && input_dims[i] > 1 {
                    shape[dim_offset] = input_dims[i] as i64;
                }
            }
            unsqueeze3_out1.expand(shape)
        };
        let reshape3_out1 = expand2_out1.reshape([1, 128, -1]);
        let slice2_out1 = reshape3_out1.slice(s![.., .., 0..100]);
        let add6_out1 = reducemean2_out1.add(slice2_out1);
        let conv1d8_out1 = self.conv1d8.forward(add6_out1);
        let relu17_out1 = burn::tensor::activation::relu(conv1d8_out1);
        let conv1d9_out1 = self.conv1d9.forward(relu17_out1);
        let sigmoid2_out1 = burn::tensor::activation::sigmoid(conv1d9_out1);
        let mul2_out1 = conv1d7_out1.mul(sigmoid2_out1);
        let concat2_out1 = burn::tensor::Tensor::cat(
            [concat1_out1, mul2_out1].into(),
            1,
        );
        let batchnormalization3_out1 = self
            .batchnormalization3
            .forward(concat2_out1.clone());
        let relu18_out1 = burn::tensor::activation::relu(batchnormalization3_out1);
        let conv1d10_out1 = self.conv1d10.forward(relu18_out1);
        let relu19_out1 = burn::tensor::activation::relu(conv1d10_out1);
        let conv1d11_out1 = self.conv1d11.forward(relu19_out1.clone());
        let reducemean3_out1 = { relu19_out1.clone().mean_dim(2usize) };
        let averagepool1d3_out1 = self.averagepool1d3.forward(relu19_out1);
        let unsqueeze4_out1: Tensor<B, 4> = averagepool1d3_out1
            .unsqueeze_dims::<4>(&[-1]);
        let expand3_out1 = {
            let onnx_shape: [i64; 4usize] = [1, 128, 1, 100];
            let input_dims = unsqueeze4_out1.dims();
            let mut shape = onnx_shape;
            #[allow(clippy::needless_range_loop)]
            for i in 0..4usize {
                let dim_offset = 4usize - 4usize + i;
                if shape[dim_offset] == 1 && input_dims[i] > 1 {
                    shape[dim_offset] = input_dims[i] as i64;
                }
            }
            unsqueeze4_out1.expand(shape)
        };
        let reshape4_out1 = expand3_out1.reshape([1, 128, -1]);
        let slice3_out1 = reshape4_out1.slice(s![.., .., 0..100]);
        let add7_out1 = reducemean3_out1.add(slice3_out1);
        let conv1d12_out1 = self.conv1d12.forward(add7_out1);
        let relu20_out1 = burn::tensor::activation::relu(conv1d12_out1);
        let conv1d13_out1 = self.conv1d13.forward(relu20_out1);
        let sigmoid3_out1 = burn::tensor::activation::sigmoid(conv1d13_out1);
        let mul3_out1 = conv1d11_out1.mul(sigmoid3_out1);
        let concat3_out1 = burn::tensor::Tensor::cat(
            [concat2_out1, mul3_out1].into(),
            1,
        );
        let batchnormalization4_out1 = self
            .batchnormalization4
            .forward(concat3_out1.clone());
        let relu21_out1 = burn::tensor::activation::relu(batchnormalization4_out1);
        let conv1d14_out1 = self.conv1d14.forward(relu21_out1);
        let relu22_out1 = burn::tensor::activation::relu(conv1d14_out1);
        let conv1d15_out1 = self.conv1d15.forward(relu22_out1.clone());
        let reducemean4_out1 = { relu22_out1.clone().mean_dim(2usize) };
        let averagepool1d4_out1 = self.averagepool1d4.forward(relu22_out1);
        let unsqueeze5_out1: Tensor<B, 4> = averagepool1d4_out1
            .unsqueeze_dims::<4>(&[-1]);
        let expand4_out1 = {
            let onnx_shape: [i64; 4usize] = [1, 128, 1, 100];
            let input_dims = unsqueeze5_out1.dims();
            let mut shape = onnx_shape;
            #[allow(clippy::needless_range_loop)]
            for i in 0..4usize {
                let dim_offset = 4usize - 4usize + i;
                if shape[dim_offset] == 1 && input_dims[i] > 1 {
                    shape[dim_offset] = input_dims[i] as i64;
                }
            }
            unsqueeze5_out1.expand(shape)
        };
        let reshape5_out1 = expand4_out1.reshape([1, 128, -1]);
        let slice4_out1 = reshape5_out1.slice(s![.., .., 0..100]);
        let add8_out1 = reducemean4_out1.add(slice4_out1);
        let conv1d16_out1 = self.conv1d16.forward(add8_out1);
        let relu23_out1 = burn::tensor::activation::relu(conv1d16_out1);
        let conv1d17_out1 = self.conv1d17.forward(relu23_out1);
        let sigmoid4_out1 = burn::tensor::activation::sigmoid(conv1d17_out1);
        let mul4_out1 = conv1d15_out1.mul(sigmoid4_out1);
        let concat4_out1 = burn::tensor::Tensor::cat(
            [concat3_out1, mul4_out1].into(),
            1,
        );
        let batchnormalization5_out1 = self
            .batchnormalization5
            .forward(concat4_out1.clone());
        let relu24_out1 = burn::tensor::activation::relu(batchnormalization5_out1);
        let conv1d18_out1 = self.conv1d18.forward(relu24_out1);
        let relu25_out1 = burn::tensor::activation::relu(conv1d18_out1);
        let conv1d19_out1 = self.conv1d19.forward(relu25_out1.clone());
        let reducemean5_out1 = { relu25_out1.clone().mean_dim(2usize) };
        let averagepool1d5_out1 = self.averagepool1d5.forward(relu25_out1);
        let unsqueeze6_out1: Tensor<B, 4> = averagepool1d5_out1
            .unsqueeze_dims::<4>(&[-1]);
        let expand5_out1 = {
            let onnx_shape: [i64; 4usize] = [1, 128, 1, 100];
            let input_dims = unsqueeze6_out1.dims();
            let mut shape = onnx_shape;
            #[allow(clippy::needless_range_loop)]
            for i in 0..4usize {
                let dim_offset = 4usize - 4usize + i;
                if shape[dim_offset] == 1 && input_dims[i] > 1 {
                    shape[dim_offset] = input_dims[i] as i64;
                }
            }
            unsqueeze6_out1.expand(shape)
        };
        let reshape6_out1 = expand5_out1.reshape([1, 128, -1]);
        let slice5_out1 = reshape6_out1.slice(s![.., .., 0..100]);
        let add9_out1 = reducemean5_out1.add(slice5_out1);
        let conv1d20_out1 = self.conv1d20.forward(add9_out1);
        let relu26_out1 = burn::tensor::activation::relu(conv1d20_out1);
        let conv1d21_out1 = self.conv1d21.forward(relu26_out1);
        let sigmoid5_out1 = burn::tensor::activation::sigmoid(conv1d21_out1);
        let mul5_out1 = conv1d19_out1.mul(sigmoid5_out1);
        let concat5_out1 = burn::tensor::Tensor::cat(
            [concat4_out1, mul5_out1].into(),
            1,
        );
        let batchnormalization6_out1 = self
            .batchnormalization6
            .forward(concat5_out1.clone());
        let relu27_out1 = burn::tensor::activation::relu(batchnormalization6_out1);
        let conv1d22_out1 = self.conv1d22.forward(relu27_out1);
        let relu28_out1 = burn::tensor::activation::relu(conv1d22_out1);
        let conv1d23_out1 = self.conv1d23.forward(relu28_out1.clone());
        let reducemean6_out1 = { relu28_out1.clone().mean_dim(2usize) };
        let averagepool1d6_out1 = self.averagepool1d6.forward(relu28_out1);
        let unsqueeze7_out1: Tensor<B, 4> = averagepool1d6_out1
            .unsqueeze_dims::<4>(&[-1]);
        let expand6_out1 = {
            let onnx_shape: [i64; 4usize] = [1, 128, 1, 100];
            let input_dims = unsqueeze7_out1.dims();
            let mut shape = onnx_shape;
            #[allow(clippy::needless_range_loop)]
            for i in 0..4usize {
                let dim_offset = 4usize - 4usize + i;
                if shape[dim_offset] == 1 && input_dims[i] > 1 {
                    shape[dim_offset] = input_dims[i] as i64;
                }
            }
            unsqueeze7_out1.expand(shape)
        };
        let reshape7_out1 = expand6_out1.reshape([1, 128, -1]);
        let slice6_out1 = reshape7_out1.slice(s![.., .., 0..100]);
        let add10_out1 = reducemean6_out1.add(slice6_out1);
        let conv1d24_out1 = self.conv1d24.forward(add10_out1);
        let relu29_out1 = burn::tensor::activation::relu(conv1d24_out1);
        let conv1d25_out1 = self.conv1d25.forward(relu29_out1);
        let sigmoid6_out1 = burn::tensor::activation::sigmoid(conv1d25_out1);
        let mul6_out1 = conv1d23_out1.mul(sigmoid6_out1);
        let concat6_out1 = burn::tensor::Tensor::cat(
            [concat5_out1, mul6_out1].into(),
            1,
        );
        let batchnormalization7_out1 = self
            .batchnormalization7
            .forward(concat6_out1.clone());
        let relu30_out1 = burn::tensor::activation::relu(batchnormalization7_out1);
        let conv1d26_out1 = self.conv1d26.forward(relu30_out1);
        let relu31_out1 = burn::tensor::activation::relu(conv1d26_out1);
        let conv1d27_out1 = self.conv1d27.forward(relu31_out1.clone());
        let reducemean7_out1 = { relu31_out1.clone().mean_dim(2usize) };
        let averagepool1d7_out1 = self.averagepool1d7.forward(relu31_out1);
        let unsqueeze8_out1: Tensor<B, 4> = averagepool1d7_out1
            .unsqueeze_dims::<4>(&[-1]);
        let expand7_out1 = {
            let onnx_shape: [i64; 4usize] = [1, 128, 1, 100];
            let input_dims = unsqueeze8_out1.dims();
            let mut shape = onnx_shape;
            #[allow(clippy::needless_range_loop)]
            for i in 0..4usize {
                let dim_offset = 4usize - 4usize + i;
                if shape[dim_offset] == 1 && input_dims[i] > 1 {
                    shape[dim_offset] = input_dims[i] as i64;
                }
            }
            unsqueeze8_out1.expand(shape)
        };
        let reshape8_out1 = expand7_out1.reshape([1, 128, -1]);
        let slice7_out1 = reshape8_out1.slice(s![.., .., 0..100]);
        let add11_out1 = reducemean7_out1.add(slice7_out1);
        let conv1d28_out1 = self.conv1d28.forward(add11_out1);
        let relu32_out1 = burn::tensor::activation::relu(conv1d28_out1);
        let conv1d29_out1 = self.conv1d29.forward(relu32_out1);
        let sigmoid7_out1 = burn::tensor::activation::sigmoid(conv1d29_out1);
        let mul7_out1 = conv1d27_out1.mul(sigmoid7_out1);
        let concat7_out1 = burn::tensor::Tensor::cat(
            [concat6_out1, mul7_out1].into(),
            1,
        );
        let batchnormalization8_out1 = self
            .batchnormalization8
            .forward(concat7_out1.clone());
        let relu33_out1 = burn::tensor::activation::relu(batchnormalization8_out1);
        let conv1d30_out1 = self.conv1d30.forward(relu33_out1);
        let relu34_out1 = burn::tensor::activation::relu(conv1d30_out1);
        let conv1d31_out1 = self.conv1d31.forward(relu34_out1.clone());
        let reducemean8_out1 = { relu34_out1.clone().mean_dim(2usize) };
        let averagepool1d8_out1 = self.averagepool1d8.forward(relu34_out1);
        let unsqueeze9_out1: Tensor<B, 4> = averagepool1d8_out1
            .unsqueeze_dims::<4>(&[-1]);
        let expand8_out1 = {
            let onnx_shape: [i64; 4usize] = [1, 128, 1, 100];
            let input_dims = unsqueeze9_out1.dims();
            let mut shape = onnx_shape;
            #[allow(clippy::needless_range_loop)]
            for i in 0..4usize {
                let dim_offset = 4usize - 4usize + i;
                if shape[dim_offset] == 1 && input_dims[i] > 1 {
                    shape[dim_offset] = input_dims[i] as i64;
                }
            }
            unsqueeze9_out1.expand(shape)
        };
        let reshape9_out1 = expand8_out1.reshape([1, 128, -1]);
        let slice8_out1 = reshape9_out1.slice(s![.., .., 0..100]);
        let add12_out1 = reducemean8_out1.add(slice8_out1);
        let conv1d32_out1 = self.conv1d32.forward(add12_out1);
        let relu35_out1 = burn::tensor::activation::relu(conv1d32_out1);
        let conv1d33_out1 = self.conv1d33.forward(relu35_out1);
        let sigmoid8_out1 = burn::tensor::activation::sigmoid(conv1d33_out1);
        let mul8_out1 = conv1d31_out1.mul(sigmoid8_out1);
        let concat8_out1 = burn::tensor::Tensor::cat(
            [concat7_out1, mul8_out1].into(),
            1,
        );
        let batchnormalization9_out1 = self
            .batchnormalization9
            .forward(concat8_out1.clone());
        let relu36_out1 = burn::tensor::activation::relu(batchnormalization9_out1);
        let conv1d34_out1 = self.conv1d34.forward(relu36_out1);
        let relu37_out1 = burn::tensor::activation::relu(conv1d34_out1);
        let conv1d35_out1 = self.conv1d35.forward(relu37_out1.clone());
        let reducemean9_out1 = { relu37_out1.clone().mean_dim(2usize) };
        let averagepool1d9_out1 = self.averagepool1d9.forward(relu37_out1);
        let unsqueeze10_out1: Tensor<B, 4> = averagepool1d9_out1
            .unsqueeze_dims::<4>(&[-1]);
        let expand9_out1 = {
            let onnx_shape: [i64; 4usize] = [1, 128, 1, 100];
            let input_dims = unsqueeze10_out1.dims();
            let mut shape = onnx_shape;
            #[allow(clippy::needless_range_loop)]
            for i in 0..4usize {
                let dim_offset = 4usize - 4usize + i;
                if shape[dim_offset] == 1 && input_dims[i] > 1 {
                    shape[dim_offset] = input_dims[i] as i64;
                }
            }
            unsqueeze10_out1.expand(shape)
        };
        let reshape10_out1 = expand9_out1.reshape([1, 128, -1]);
        let slice9_out1 = reshape10_out1.slice(s![.., .., 0..100]);
        let add13_out1 = reducemean9_out1.add(slice9_out1);
        let conv1d36_out1 = self.conv1d36.forward(add13_out1);
        let relu38_out1 = burn::tensor::activation::relu(conv1d36_out1);
        let conv1d37_out1 = self.conv1d37.forward(relu38_out1);
        let sigmoid9_out1 = burn::tensor::activation::sigmoid(conv1d37_out1);
        let mul9_out1 = conv1d35_out1.mul(sigmoid9_out1);
        let concat9_out1 = burn::tensor::Tensor::cat(
            [concat8_out1, mul9_out1].into(),
            1,
        );
        let batchnormalization10_out1 = self
            .batchnormalization10
            .forward(concat9_out1.clone());
        let relu39_out1 = burn::tensor::activation::relu(batchnormalization10_out1);
        let conv1d38_out1 = self.conv1d38.forward(relu39_out1);
        let relu40_out1 = burn::tensor::activation::relu(conv1d38_out1);
        let conv1d39_out1 = self.conv1d39.forward(relu40_out1.clone());
        let reducemean10_out1 = { relu40_out1.clone().mean_dim(2usize) };
        let averagepool1d10_out1 = self.averagepool1d10.forward(relu40_out1);
        let unsqueeze11_out1: Tensor<B, 4> = averagepool1d10_out1
            .unsqueeze_dims::<4>(&[-1]);
        let expand10_out1 = {
            let onnx_shape: [i64; 4usize] = [1, 128, 1, 100];
            let input_dims = unsqueeze11_out1.dims();
            let mut shape = onnx_shape;
            #[allow(clippy::needless_range_loop)]
            for i in 0..4usize {
                let dim_offset = 4usize - 4usize + i;
                if shape[dim_offset] == 1 && input_dims[i] > 1 {
                    shape[dim_offset] = input_dims[i] as i64;
                }
            }
            unsqueeze11_out1.expand(shape)
        };
        let reshape11_out1 = expand10_out1.reshape([1, 128, -1]);
        let slice10_out1 = reshape11_out1.slice(s![.., .., 0..100]);
        let add14_out1 = reducemean10_out1.add(slice10_out1);
        let conv1d40_out1 = self.conv1d40.forward(add14_out1);
        let relu41_out1 = burn::tensor::activation::relu(conv1d40_out1);
        let conv1d41_out1 = self.conv1d41.forward(relu41_out1);
        let sigmoid10_out1 = burn::tensor::activation::sigmoid(conv1d41_out1);
        let mul10_out1 = conv1d39_out1.mul(sigmoid10_out1);
        let concat10_out1 = burn::tensor::Tensor::cat(
            [concat9_out1, mul10_out1].into(),
            1,
        );
        let batchnormalization11_out1 = self
            .batchnormalization11
            .forward(concat10_out1.clone());
        let relu42_out1 = burn::tensor::activation::relu(batchnormalization11_out1);
        let conv1d42_out1 = self.conv1d42.forward(relu42_out1);
        let relu43_out1 = burn::tensor::activation::relu(conv1d42_out1);
        let conv1d43_out1 = self.conv1d43.forward(relu43_out1.clone());
        let reducemean11_out1 = { relu43_out1.clone().mean_dim(2usize) };
        let averagepool1d11_out1 = self.averagepool1d11.forward(relu43_out1);
        let unsqueeze12_out1: Tensor<B, 4> = averagepool1d11_out1
            .unsqueeze_dims::<4>(&[-1]);
        let expand11_out1 = {
            let onnx_shape: [i64; 4usize] = [1, 128, 1, 100];
            let input_dims = unsqueeze12_out1.dims();
            let mut shape = onnx_shape;
            #[allow(clippy::needless_range_loop)]
            for i in 0..4usize {
                let dim_offset = 4usize - 4usize + i;
                if shape[dim_offset] == 1 && input_dims[i] > 1 {
                    shape[dim_offset] = input_dims[i] as i64;
                }
            }
            unsqueeze12_out1.expand(shape)
        };
        let reshape12_out1 = expand11_out1.reshape([1, 128, -1]);
        let slice11_out1 = reshape12_out1.slice(s![.., .., 0..100]);
        let add15_out1 = reducemean11_out1.add(slice11_out1);
        let conv1d44_out1 = self.conv1d44.forward(add15_out1);
        let relu44_out1 = burn::tensor::activation::relu(conv1d44_out1);
        let conv1d45_out1 = self.conv1d45.forward(relu44_out1);
        let sigmoid11_out1 = burn::tensor::activation::sigmoid(conv1d45_out1);
        let mul11_out1 = conv1d43_out1.mul(sigmoid11_out1);
        let concat11_out1 = burn::tensor::Tensor::cat(
            [concat10_out1, mul11_out1].into(),
            1,
        );
        let batchnormalization12_out1 = self
            .batchnormalization12
            .forward(concat11_out1.clone());
        let relu45_out1 = burn::tensor::activation::relu(batchnormalization12_out1);
        let conv1d46_out1 = self.conv1d46.forward(relu45_out1);
        let relu46_out1 = burn::tensor::activation::relu(conv1d46_out1);
        let conv1d47_out1 = self.conv1d47.forward(relu46_out1.clone());
        let reducemean12_out1 = { relu46_out1.clone().mean_dim(2usize) };
        let averagepool1d12_out1 = self.averagepool1d12.forward(relu46_out1);
        let unsqueeze13_out1: Tensor<B, 4> = averagepool1d12_out1
            .unsqueeze_dims::<4>(&[-1]);
        let expand12_out1 = {
            let onnx_shape: [i64; 4usize] = [1, 128, 1, 100];
            let input_dims = unsqueeze13_out1.dims();
            let mut shape = onnx_shape;
            #[allow(clippy::needless_range_loop)]
            for i in 0..4usize {
                let dim_offset = 4usize - 4usize + i;
                if shape[dim_offset] == 1 && input_dims[i] > 1 {
                    shape[dim_offset] = input_dims[i] as i64;
                }
            }
            unsqueeze13_out1.expand(shape)
        };
        let reshape13_out1 = expand12_out1.reshape([1, 128, -1]);
        let slice12_out1 = reshape13_out1.slice(s![.., .., 0..100]);
        let add16_out1 = reducemean12_out1.add(slice12_out1);
        let conv1d48_out1 = self.conv1d48.forward(add16_out1);
        let relu47_out1 = burn::tensor::activation::relu(conv1d48_out1);
        let conv1d49_out1 = self.conv1d49.forward(relu47_out1);
        let sigmoid12_out1 = burn::tensor::activation::sigmoid(conv1d49_out1);
        let mul12_out1 = conv1d47_out1.mul(sigmoid12_out1);
        let concat12_out1 = burn::tensor::Tensor::cat(
            [concat11_out1, mul12_out1].into(),
            1,
        );
        let batchnormalization13_out1 = self.batchnormalization13.forward(concat12_out1);
        let relu48_out1 = burn::tensor::activation::relu(batchnormalization13_out1);
        let conv1d50_out1 = self.conv1d50.forward(relu48_out1);
        let batchnormalization14_out1 = self
            .batchnormalization14
            .forward(conv1d50_out1.clone());
        let relu49_out1 = burn::tensor::activation::relu(batchnormalization14_out1);
        let conv1d51_out1 = self.conv1d51.forward(relu49_out1);
        let relu50_out1 = burn::tensor::activation::relu(conv1d51_out1);
        let conv1d52_out1 = self.conv1d52.forward(relu50_out1.clone());
        let reducemean13_out1 = { relu50_out1.clone().mean_dim(2usize) };
        let averagepool1d13_out1 = self.averagepool1d13.forward(relu50_out1);
        let unsqueeze14_out1: Tensor<B, 4> = averagepool1d13_out1
            .unsqueeze_dims::<4>(&[-1]);
        let expand13_out1 = {
            let onnx_shape: [i64; 4usize] = [1, 128, 1, 100];
            let input_dims = unsqueeze14_out1.dims();
            let mut shape = onnx_shape;
            #[allow(clippy::needless_range_loop)]
            for i in 0..4usize {
                let dim_offset = 4usize - 4usize + i;
                if shape[dim_offset] == 1 && input_dims[i] > 1 {
                    shape[dim_offset] = input_dims[i] as i64;
                }
            }
            unsqueeze14_out1.expand(shape)
        };
        let reshape14_out1 = expand13_out1.reshape([1, 128, -1]);
        let slice13_out1 = reshape14_out1.slice(s![.., .., 0..100]);
        let add17_out1 = reducemean13_out1.add(slice13_out1);
        let conv1d53_out1 = self.conv1d53.forward(add17_out1);
        let relu51_out1 = burn::tensor::activation::relu(conv1d53_out1);
        let conv1d54_out1 = self.conv1d54.forward(relu51_out1);
        let sigmoid13_out1 = burn::tensor::activation::sigmoid(conv1d54_out1);
        let mul13_out1 = conv1d52_out1.mul(sigmoid13_out1);
        let concat13_out1 = burn::tensor::Tensor::cat(
            [conv1d50_out1, mul13_out1].into(),
            1,
        );
        let batchnormalization15_out1 = self
            .batchnormalization15
            .forward(concat13_out1.clone());
        let relu52_out1 = burn::tensor::activation::relu(batchnormalization15_out1);
        let conv1d55_out1 = self.conv1d55.forward(relu52_out1);
        let relu53_out1 = burn::tensor::activation::relu(conv1d55_out1);
        let conv1d56_out1 = self.conv1d56.forward(relu53_out1.clone());
        let reducemean14_out1 = { relu53_out1.clone().mean_dim(2usize) };
        let averagepool1d14_out1 = self.averagepool1d14.forward(relu53_out1);
        let unsqueeze15_out1: Tensor<B, 4> = averagepool1d14_out1
            .unsqueeze_dims::<4>(&[-1]);
        let expand14_out1 = {
            let onnx_shape: [i64; 4usize] = [1, 128, 1, 100];
            let input_dims = unsqueeze15_out1.dims();
            let mut shape = onnx_shape;
            #[allow(clippy::needless_range_loop)]
            for i in 0..4usize {
                let dim_offset = 4usize - 4usize + i;
                if shape[dim_offset] == 1 && input_dims[i] > 1 {
                    shape[dim_offset] = input_dims[i] as i64;
                }
            }
            unsqueeze15_out1.expand(shape)
        };
        let reshape15_out1 = expand14_out1.reshape([1, 128, -1]);
        let slice14_out1 = reshape15_out1.slice(s![.., .., 0..100]);
        let add18_out1 = reducemean14_out1.add(slice14_out1);
        let conv1d57_out1 = self.conv1d57.forward(add18_out1);
        let relu54_out1 = burn::tensor::activation::relu(conv1d57_out1);
        let conv1d58_out1 = self.conv1d58.forward(relu54_out1);
        let sigmoid14_out1 = burn::tensor::activation::sigmoid(conv1d58_out1);
        let mul14_out1 = conv1d56_out1.mul(sigmoid14_out1);
        let concat14_out1 = burn::tensor::Tensor::cat(
            [concat13_out1, mul14_out1].into(),
            1,
        );
        let batchnormalization16_out1 = self
            .batchnormalization16
            .forward(concat14_out1.clone());
        let relu55_out1 = burn::tensor::activation::relu(batchnormalization16_out1);
        let conv1d59_out1 = self.conv1d59.forward(relu55_out1);
        let relu56_out1 = burn::tensor::activation::relu(conv1d59_out1);
        let conv1d60_out1 = self.conv1d60.forward(relu56_out1.clone());
        let reducemean15_out1 = { relu56_out1.clone().mean_dim(2usize) };
        let averagepool1d15_out1 = self.averagepool1d15.forward(relu56_out1);
        let unsqueeze16_out1: Tensor<B, 4> = averagepool1d15_out1
            .unsqueeze_dims::<4>(&[-1]);
        let expand15_out1 = {
            let onnx_shape: [i64; 4usize] = [1, 128, 1, 100];
            let input_dims = unsqueeze16_out1.dims();
            let mut shape = onnx_shape;
            #[allow(clippy::needless_range_loop)]
            for i in 0..4usize {
                let dim_offset = 4usize - 4usize + i;
                if shape[dim_offset] == 1 && input_dims[i] > 1 {
                    shape[dim_offset] = input_dims[i] as i64;
                }
            }
            unsqueeze16_out1.expand(shape)
        };
        let reshape16_out1 = expand15_out1.reshape([1, 128, -1]);
        let slice15_out1 = reshape16_out1.slice(s![.., .., 0..100]);
        let add19_out1 = reducemean15_out1.add(slice15_out1);
        let conv1d61_out1 = self.conv1d61.forward(add19_out1);
        let relu57_out1 = burn::tensor::activation::relu(conv1d61_out1);
        let conv1d62_out1 = self.conv1d62.forward(relu57_out1);
        let sigmoid15_out1 = burn::tensor::activation::sigmoid(conv1d62_out1);
        let mul15_out1 = conv1d60_out1.mul(sigmoid15_out1);
        let concat15_out1 = burn::tensor::Tensor::cat(
            [concat14_out1, mul15_out1].into(),
            1,
        );
        let batchnormalization17_out1 = self
            .batchnormalization17
            .forward(concat15_out1.clone());
        let relu58_out1 = burn::tensor::activation::relu(batchnormalization17_out1);
        let conv1d63_out1 = self.conv1d63.forward(relu58_out1);
        let relu59_out1 = burn::tensor::activation::relu(conv1d63_out1);
        let conv1d64_out1 = self.conv1d64.forward(relu59_out1.clone());
        let reducemean16_out1 = { relu59_out1.clone().mean_dim(2usize) };
        let averagepool1d16_out1 = self.averagepool1d16.forward(relu59_out1);
        let unsqueeze17_out1: Tensor<B, 4> = averagepool1d16_out1
            .unsqueeze_dims::<4>(&[-1]);
        let expand16_out1 = {
            let onnx_shape: [i64; 4usize] = [1, 128, 1, 100];
            let input_dims = unsqueeze17_out1.dims();
            let mut shape = onnx_shape;
            #[allow(clippy::needless_range_loop)]
            for i in 0..4usize {
                let dim_offset = 4usize - 4usize + i;
                if shape[dim_offset] == 1 && input_dims[i] > 1 {
                    shape[dim_offset] = input_dims[i] as i64;
                }
            }
            unsqueeze17_out1.expand(shape)
        };
        let reshape17_out1 = expand16_out1.reshape([1, 128, -1]);
        let slice16_out1 = reshape17_out1.slice(s![.., .., 0..100]);
        let add20_out1 = reducemean16_out1.add(slice16_out1);
        let conv1d65_out1 = self.conv1d65.forward(add20_out1);
        let relu60_out1 = burn::tensor::activation::relu(conv1d65_out1);
        let conv1d66_out1 = self.conv1d66.forward(relu60_out1);
        let sigmoid16_out1 = burn::tensor::activation::sigmoid(conv1d66_out1);
        let mul16_out1 = conv1d64_out1.mul(sigmoid16_out1);
        let concat16_out1 = burn::tensor::Tensor::cat(
            [concat15_out1, mul16_out1].into(),
            1,
        );
        let batchnormalization18_out1 = self
            .batchnormalization18
            .forward(concat16_out1.clone());
        let relu61_out1 = burn::tensor::activation::relu(batchnormalization18_out1);
        let conv1d67_out1 = self.conv1d67.forward(relu61_out1);
        let relu62_out1 = burn::tensor::activation::relu(conv1d67_out1);
        let conv1d68_out1 = self.conv1d68.forward(relu62_out1.clone());
        let reducemean17_out1 = { relu62_out1.clone().mean_dim(2usize) };
        let averagepool1d17_out1 = self.averagepool1d17.forward(relu62_out1);
        let unsqueeze18_out1: Tensor<B, 4> = averagepool1d17_out1
            .unsqueeze_dims::<4>(&[-1]);
        let expand17_out1 = {
            let onnx_shape: [i64; 4usize] = [1, 128, 1, 100];
            let input_dims = unsqueeze18_out1.dims();
            let mut shape = onnx_shape;
            #[allow(clippy::needless_range_loop)]
            for i in 0..4usize {
                let dim_offset = 4usize - 4usize + i;
                if shape[dim_offset] == 1 && input_dims[i] > 1 {
                    shape[dim_offset] = input_dims[i] as i64;
                }
            }
            unsqueeze18_out1.expand(shape)
        };
        let reshape18_out1 = expand17_out1.reshape([1, 128, -1]);
        let slice17_out1 = reshape18_out1.slice(s![.., .., 0..100]);
        let add21_out1 = reducemean17_out1.add(slice17_out1);
        let conv1d69_out1 = self.conv1d69.forward(add21_out1);
        let relu63_out1 = burn::tensor::activation::relu(conv1d69_out1);
        let conv1d70_out1 = self.conv1d70.forward(relu63_out1);
        let sigmoid17_out1 = burn::tensor::activation::sigmoid(conv1d70_out1);
        let mul17_out1 = conv1d68_out1.mul(sigmoid17_out1);
        let concat17_out1 = burn::tensor::Tensor::cat(
            [concat16_out1, mul17_out1].into(),
            1,
        );
        let batchnormalization19_out1 = self
            .batchnormalization19
            .forward(concat17_out1.clone());
        let relu64_out1 = burn::tensor::activation::relu(batchnormalization19_out1);
        let conv1d71_out1 = self.conv1d71.forward(relu64_out1);
        let relu65_out1 = burn::tensor::activation::relu(conv1d71_out1);
        let conv1d72_out1 = self.conv1d72.forward(relu65_out1.clone());
        let reducemean18_out1 = { relu65_out1.clone().mean_dim(2usize) };
        let averagepool1d18_out1 = self.averagepool1d18.forward(relu65_out1);
        let unsqueeze19_out1: Tensor<B, 4> = averagepool1d18_out1
            .unsqueeze_dims::<4>(&[-1]);
        let expand18_out1 = {
            let onnx_shape: [i64; 4usize] = [1, 128, 1, 100];
            let input_dims = unsqueeze19_out1.dims();
            let mut shape = onnx_shape;
            #[allow(clippy::needless_range_loop)]
            for i in 0..4usize {
                let dim_offset = 4usize - 4usize + i;
                if shape[dim_offset] == 1 && input_dims[i] > 1 {
                    shape[dim_offset] = input_dims[i] as i64;
                }
            }
            unsqueeze19_out1.expand(shape)
        };
        let reshape19_out1 = expand18_out1.reshape([1, 128, -1]);
        let slice18_out1 = reshape19_out1.slice(s![.., .., 0..100]);
        let add22_out1 = reducemean18_out1.add(slice18_out1);
        let conv1d73_out1 = self.conv1d73.forward(add22_out1);
        let relu66_out1 = burn::tensor::activation::relu(conv1d73_out1);
        let conv1d74_out1 = self.conv1d74.forward(relu66_out1);
        let sigmoid18_out1 = burn::tensor::activation::sigmoid(conv1d74_out1);
        let mul18_out1 = conv1d72_out1.mul(sigmoid18_out1);
        let concat18_out1 = burn::tensor::Tensor::cat(
            [concat17_out1, mul18_out1].into(),
            1,
        );
        let batchnormalization20_out1 = self
            .batchnormalization20
            .forward(concat18_out1.clone());
        let relu67_out1 = burn::tensor::activation::relu(batchnormalization20_out1);
        let conv1d75_out1 = self.conv1d75.forward(relu67_out1);
        let relu68_out1 = burn::tensor::activation::relu(conv1d75_out1);
        let conv1d76_out1 = self.conv1d76.forward(relu68_out1.clone());
        let reducemean19_out1 = { relu68_out1.clone().mean_dim(2usize) };
        let averagepool1d19_out1 = self.averagepool1d19.forward(relu68_out1);
        let unsqueeze20_out1: Tensor<B, 4> = averagepool1d19_out1
            .unsqueeze_dims::<4>(&[-1]);
        let expand19_out1 = {
            let onnx_shape: [i64; 4usize] = [1, 128, 1, 100];
            let input_dims = unsqueeze20_out1.dims();
            let mut shape = onnx_shape;
            #[allow(clippy::needless_range_loop)]
            for i in 0..4usize {
                let dim_offset = 4usize - 4usize + i;
                if shape[dim_offset] == 1 && input_dims[i] > 1 {
                    shape[dim_offset] = input_dims[i] as i64;
                }
            }
            unsqueeze20_out1.expand(shape)
        };
        let reshape20_out1 = expand19_out1.reshape([1, 128, -1]);
        let slice19_out1 = reshape20_out1.slice(s![.., .., 0..100]);
        let add23_out1 = reducemean19_out1.add(slice19_out1);
        let conv1d77_out1 = self.conv1d77.forward(add23_out1);
        let relu69_out1 = burn::tensor::activation::relu(conv1d77_out1);
        let conv1d78_out1 = self.conv1d78.forward(relu69_out1);
        let sigmoid19_out1 = burn::tensor::activation::sigmoid(conv1d78_out1);
        let mul19_out1 = conv1d76_out1.mul(sigmoid19_out1);
        let concat19_out1 = burn::tensor::Tensor::cat(
            [concat18_out1, mul19_out1].into(),
            1,
        );
        let batchnormalization21_out1 = self
            .batchnormalization21
            .forward(concat19_out1.clone());
        let relu70_out1 = burn::tensor::activation::relu(batchnormalization21_out1);
        let conv1d79_out1 = self.conv1d79.forward(relu70_out1);
        let relu71_out1 = burn::tensor::activation::relu(conv1d79_out1);
        let conv1d80_out1 = self.conv1d80.forward(relu71_out1.clone());
        let reducemean20_out1 = { relu71_out1.clone().mean_dim(2usize) };
        let averagepool1d20_out1 = self.averagepool1d20.forward(relu71_out1);
        let unsqueeze21_out1: Tensor<B, 4> = averagepool1d20_out1
            .unsqueeze_dims::<4>(&[-1]);
        let expand20_out1 = {
            let onnx_shape: [i64; 4usize] = [1, 128, 1, 100];
            let input_dims = unsqueeze21_out1.dims();
            let mut shape = onnx_shape;
            #[allow(clippy::needless_range_loop)]
            for i in 0..4usize {
                let dim_offset = 4usize - 4usize + i;
                if shape[dim_offset] == 1 && input_dims[i] > 1 {
                    shape[dim_offset] = input_dims[i] as i64;
                }
            }
            unsqueeze21_out1.expand(shape)
        };
        let reshape21_out1 = expand20_out1.reshape([1, 128, -1]);
        let slice20_out1 = reshape21_out1.slice(s![.., .., 0..100]);
        let add24_out1 = reducemean20_out1.add(slice20_out1);
        let conv1d81_out1 = self.conv1d81.forward(add24_out1);
        let relu72_out1 = burn::tensor::activation::relu(conv1d81_out1);
        let conv1d82_out1 = self.conv1d82.forward(relu72_out1);
        let sigmoid20_out1 = burn::tensor::activation::sigmoid(conv1d82_out1);
        let mul20_out1 = conv1d80_out1.mul(sigmoid20_out1);
        let concat20_out1 = burn::tensor::Tensor::cat(
            [concat19_out1, mul20_out1].into(),
            1,
        );
        let batchnormalization22_out1 = self
            .batchnormalization22
            .forward(concat20_out1.clone());
        let relu73_out1 = burn::tensor::activation::relu(batchnormalization22_out1);
        let conv1d83_out1 = self.conv1d83.forward(relu73_out1);
        let relu74_out1 = burn::tensor::activation::relu(conv1d83_out1);
        let conv1d84_out1 = self.conv1d84.forward(relu74_out1.clone());
        let reducemean21_out1 = { relu74_out1.clone().mean_dim(2usize) };
        let averagepool1d21_out1 = self.averagepool1d21.forward(relu74_out1);
        let unsqueeze22_out1: Tensor<B, 4> = averagepool1d21_out1
            .unsqueeze_dims::<4>(&[-1]);
        let expand21_out1 = {
            let onnx_shape: [i64; 4usize] = [1, 128, 1, 100];
            let input_dims = unsqueeze22_out1.dims();
            let mut shape = onnx_shape;
            #[allow(clippy::needless_range_loop)]
            for i in 0..4usize {
                let dim_offset = 4usize - 4usize + i;
                if shape[dim_offset] == 1 && input_dims[i] > 1 {
                    shape[dim_offset] = input_dims[i] as i64;
                }
            }
            unsqueeze22_out1.expand(shape)
        };
        let reshape22_out1 = expand21_out1.reshape([1, 128, -1]);
        let slice21_out1 = reshape22_out1.slice(s![.., .., 0..100]);
        let add25_out1 = reducemean21_out1.add(slice21_out1);
        let conv1d85_out1 = self.conv1d85.forward(add25_out1);
        let relu75_out1 = burn::tensor::activation::relu(conv1d85_out1);
        let conv1d86_out1 = self.conv1d86.forward(relu75_out1);
        let sigmoid21_out1 = burn::tensor::activation::sigmoid(conv1d86_out1);
        let mul21_out1 = conv1d84_out1.mul(sigmoid21_out1);
        let concat21_out1 = burn::tensor::Tensor::cat(
            [concat20_out1, mul21_out1].into(),
            1,
        );
        let batchnormalization23_out1 = self
            .batchnormalization23
            .forward(concat21_out1.clone());
        let relu76_out1 = burn::tensor::activation::relu(batchnormalization23_out1);
        let conv1d87_out1 = self.conv1d87.forward(relu76_out1);
        let relu77_out1 = burn::tensor::activation::relu(conv1d87_out1);
        let conv1d88_out1 = self.conv1d88.forward(relu77_out1.clone());
        let reducemean22_out1 = { relu77_out1.clone().mean_dim(2usize) };
        let averagepool1d22_out1 = self.averagepool1d22.forward(relu77_out1);
        let unsqueeze23_out1: Tensor<B, 4> = averagepool1d22_out1
            .unsqueeze_dims::<4>(&[-1]);
        let expand22_out1 = {
            let onnx_shape: [i64; 4usize] = [1, 128, 1, 100];
            let input_dims = unsqueeze23_out1.dims();
            let mut shape = onnx_shape;
            #[allow(clippy::needless_range_loop)]
            for i in 0..4usize {
                let dim_offset = 4usize - 4usize + i;
                if shape[dim_offset] == 1 && input_dims[i] > 1 {
                    shape[dim_offset] = input_dims[i] as i64;
                }
            }
            unsqueeze23_out1.expand(shape)
        };
        let reshape23_out1 = expand22_out1.reshape([1, 128, -1]);
        let slice22_out1 = reshape23_out1.slice(s![.., .., 0..100]);
        let add26_out1 = reducemean22_out1.add(slice22_out1);
        let conv1d89_out1 = self.conv1d89.forward(add26_out1);
        let relu78_out1 = burn::tensor::activation::relu(conv1d89_out1);
        let conv1d90_out1 = self.conv1d90.forward(relu78_out1);
        let sigmoid22_out1 = burn::tensor::activation::sigmoid(conv1d90_out1);
        let mul22_out1 = conv1d88_out1.mul(sigmoid22_out1);
        let concat22_out1 = burn::tensor::Tensor::cat(
            [concat21_out1, mul22_out1].into(),
            1,
        );
        let batchnormalization24_out1 = self
            .batchnormalization24
            .forward(concat22_out1.clone());
        let relu79_out1 = burn::tensor::activation::relu(batchnormalization24_out1);
        let conv1d91_out1 = self.conv1d91.forward(relu79_out1);
        let relu80_out1 = burn::tensor::activation::relu(conv1d91_out1);
        let conv1d92_out1 = self.conv1d92.forward(relu80_out1.clone());
        let reducemean23_out1 = { relu80_out1.clone().mean_dim(2usize) };
        let averagepool1d23_out1 = self.averagepool1d23.forward(relu80_out1);
        let unsqueeze24_out1: Tensor<B, 4> = averagepool1d23_out1
            .unsqueeze_dims::<4>(&[-1]);
        let expand23_out1 = {
            let onnx_shape: [i64; 4usize] = [1, 128, 1, 100];
            let input_dims = unsqueeze24_out1.dims();
            let mut shape = onnx_shape;
            #[allow(clippy::needless_range_loop)]
            for i in 0..4usize {
                let dim_offset = 4usize - 4usize + i;
                if shape[dim_offset] == 1 && input_dims[i] > 1 {
                    shape[dim_offset] = input_dims[i] as i64;
                }
            }
            unsqueeze24_out1.expand(shape)
        };
        let reshape24_out1 = expand23_out1.reshape([1, 128, -1]);
        let slice23_out1 = reshape24_out1.slice(s![.., .., 0..100]);
        let add27_out1 = reducemean23_out1.add(slice23_out1);
        let conv1d93_out1 = self.conv1d93.forward(add27_out1);
        let relu81_out1 = burn::tensor::activation::relu(conv1d93_out1);
        let conv1d94_out1 = self.conv1d94.forward(relu81_out1);
        let sigmoid23_out1 = burn::tensor::activation::sigmoid(conv1d94_out1);
        let mul23_out1 = conv1d92_out1.mul(sigmoid23_out1);
        let concat23_out1 = burn::tensor::Tensor::cat(
            [concat22_out1, mul23_out1].into(),
            1,
        );
        let batchnormalization25_out1 = self
            .batchnormalization25
            .forward(concat23_out1.clone());
        let relu82_out1 = burn::tensor::activation::relu(batchnormalization25_out1);
        let conv1d95_out1 = self.conv1d95.forward(relu82_out1);
        let relu83_out1 = burn::tensor::activation::relu(conv1d95_out1);
        let conv1d96_out1 = self.conv1d96.forward(relu83_out1.clone());
        let reducemean24_out1 = { relu83_out1.clone().mean_dim(2usize) };
        let averagepool1d24_out1 = self.averagepool1d24.forward(relu83_out1);
        let unsqueeze25_out1: Tensor<B, 4> = averagepool1d24_out1
            .unsqueeze_dims::<4>(&[-1]);
        let expand24_out1 = {
            let onnx_shape: [i64; 4usize] = [1, 128, 1, 100];
            let input_dims = unsqueeze25_out1.dims();
            let mut shape = onnx_shape;
            #[allow(clippy::needless_range_loop)]
            for i in 0..4usize {
                let dim_offset = 4usize - 4usize + i;
                if shape[dim_offset] == 1 && input_dims[i] > 1 {
                    shape[dim_offset] = input_dims[i] as i64;
                }
            }
            unsqueeze25_out1.expand(shape)
        };
        let reshape25_out1 = expand24_out1.reshape([1, 128, -1]);
        let slice24_out1 = reshape25_out1.slice(s![.., .., 0..100]);
        let add28_out1 = reducemean24_out1.add(slice24_out1);
        let conv1d97_out1 = self.conv1d97.forward(add28_out1);
        let relu84_out1 = burn::tensor::activation::relu(conv1d97_out1);
        let conv1d98_out1 = self.conv1d98.forward(relu84_out1);
        let sigmoid24_out1 = burn::tensor::activation::sigmoid(conv1d98_out1);
        let mul24_out1 = conv1d96_out1.mul(sigmoid24_out1);
        let concat24_out1 = burn::tensor::Tensor::cat(
            [concat23_out1, mul24_out1].into(),
            1,
        );
        let batchnormalization26_out1 = self
            .batchnormalization26
            .forward(concat24_out1.clone());
        let relu85_out1 = burn::tensor::activation::relu(batchnormalization26_out1);
        let conv1d99_out1 = self.conv1d99.forward(relu85_out1);
        let relu86_out1 = burn::tensor::activation::relu(conv1d99_out1);
        let conv1d100_out1 = self.conv1d100.forward(relu86_out1.clone());
        let reducemean25_out1 = { relu86_out1.clone().mean_dim(2usize) };
        let averagepool1d25_out1 = self.averagepool1d25.forward(relu86_out1);
        let unsqueeze26_out1: Tensor<B, 4> = averagepool1d25_out1
            .unsqueeze_dims::<4>(&[-1]);
        let expand25_out1 = {
            let onnx_shape: [i64; 4usize] = [1, 128, 1, 100];
            let input_dims = unsqueeze26_out1.dims();
            let mut shape = onnx_shape;
            #[allow(clippy::needless_range_loop)]
            for i in 0..4usize {
                let dim_offset = 4usize - 4usize + i;
                if shape[dim_offset] == 1 && input_dims[i] > 1 {
                    shape[dim_offset] = input_dims[i] as i64;
                }
            }
            unsqueeze26_out1.expand(shape)
        };
        let reshape26_out1 = expand25_out1.reshape([1, 128, -1]);
        let slice25_out1 = reshape26_out1.slice(s![.., .., 0..100]);
        let add29_out1 = reducemean25_out1.add(slice25_out1);
        let conv1d101_out1 = self.conv1d101.forward(add29_out1);
        let relu87_out1 = burn::tensor::activation::relu(conv1d101_out1);
        let conv1d102_out1 = self.conv1d102.forward(relu87_out1);
        let sigmoid25_out1 = burn::tensor::activation::sigmoid(conv1d102_out1);
        let mul25_out1 = conv1d100_out1.mul(sigmoid25_out1);
        let concat25_out1 = burn::tensor::Tensor::cat(
            [concat24_out1, mul25_out1].into(),
            1,
        );
        let batchnormalization27_out1 = self
            .batchnormalization27
            .forward(concat25_out1.clone());
        let relu88_out1 = burn::tensor::activation::relu(batchnormalization27_out1);
        let conv1d103_out1 = self.conv1d103.forward(relu88_out1);
        let relu89_out1 = burn::tensor::activation::relu(conv1d103_out1);
        let conv1d104_out1 = self.conv1d104.forward(relu89_out1.clone());
        let reducemean26_out1 = { relu89_out1.clone().mean_dim(2usize) };
        let averagepool1d26_out1 = self.averagepool1d26.forward(relu89_out1);
        let unsqueeze27_out1: Tensor<B, 4> = averagepool1d26_out1
            .unsqueeze_dims::<4>(&[-1]);
        let expand26_out1 = {
            let onnx_shape: [i64; 4usize] = [1, 128, 1, 100];
            let input_dims = unsqueeze27_out1.dims();
            let mut shape = onnx_shape;
            #[allow(clippy::needless_range_loop)]
            for i in 0..4usize {
                let dim_offset = 4usize - 4usize + i;
                if shape[dim_offset] == 1 && input_dims[i] > 1 {
                    shape[dim_offset] = input_dims[i] as i64;
                }
            }
            unsqueeze27_out1.expand(shape)
        };
        let reshape27_out1 = expand26_out1.reshape([1, 128, -1]);
        let slice26_out1 = reshape27_out1.slice(s![.., .., 0..100]);
        let add30_out1 = reducemean26_out1.add(slice26_out1);
        let conv1d105_out1 = self.conv1d105.forward(add30_out1);
        let relu90_out1 = burn::tensor::activation::relu(conv1d105_out1);
        let conv1d106_out1 = self.conv1d106.forward(relu90_out1);
        let sigmoid26_out1 = burn::tensor::activation::sigmoid(conv1d106_out1);
        let mul26_out1 = conv1d104_out1.mul(sigmoid26_out1);
        let concat26_out1 = burn::tensor::Tensor::cat(
            [concat25_out1, mul26_out1].into(),
            1,
        );
        let batchnormalization28_out1 = self
            .batchnormalization28
            .forward(concat26_out1.clone());
        let relu91_out1 = burn::tensor::activation::relu(batchnormalization28_out1);
        let conv1d107_out1 = self.conv1d107.forward(relu91_out1);
        let relu92_out1 = burn::tensor::activation::relu(conv1d107_out1);
        let conv1d108_out1 = self.conv1d108.forward(relu92_out1.clone());
        let reducemean27_out1 = { relu92_out1.clone().mean_dim(2usize) };
        let averagepool1d27_out1 = self.averagepool1d27.forward(relu92_out1);
        let unsqueeze28_out1: Tensor<B, 4> = averagepool1d27_out1
            .unsqueeze_dims::<4>(&[-1]);
        let expand27_out1 = {
            let onnx_shape: [i64; 4usize] = [1, 128, 1, 100];
            let input_dims = unsqueeze28_out1.dims();
            let mut shape = onnx_shape;
            #[allow(clippy::needless_range_loop)]
            for i in 0..4usize {
                let dim_offset = 4usize - 4usize + i;
                if shape[dim_offset] == 1 && input_dims[i] > 1 {
                    shape[dim_offset] = input_dims[i] as i64;
                }
            }
            unsqueeze28_out1.expand(shape)
        };
        let reshape28_out1 = expand27_out1.reshape([1, 128, -1]);
        let slice27_out1 = reshape28_out1.slice(s![.., .., 0..100]);
        let add31_out1 = reducemean27_out1.add(slice27_out1);
        let conv1d109_out1 = self.conv1d109.forward(add31_out1);
        let relu93_out1 = burn::tensor::activation::relu(conv1d109_out1);
        let conv1d110_out1 = self.conv1d110.forward(relu93_out1);
        let sigmoid27_out1 = burn::tensor::activation::sigmoid(conv1d110_out1);
        let mul27_out1 = conv1d108_out1.mul(sigmoid27_out1);
        let concat27_out1 = burn::tensor::Tensor::cat(
            [concat26_out1, mul27_out1].into(),
            1,
        );
        let batchnormalization29_out1 = self
            .batchnormalization29
            .forward(concat27_out1.clone());
        let relu94_out1 = burn::tensor::activation::relu(batchnormalization29_out1);
        let conv1d111_out1 = self.conv1d111.forward(relu94_out1);
        let relu95_out1 = burn::tensor::activation::relu(conv1d111_out1);
        let conv1d112_out1 = self.conv1d112.forward(relu95_out1.clone());
        let reducemean28_out1 = { relu95_out1.clone().mean_dim(2usize) };
        let averagepool1d28_out1 = self.averagepool1d28.forward(relu95_out1);
        let unsqueeze29_out1: Tensor<B, 4> = averagepool1d28_out1
            .unsqueeze_dims::<4>(&[-1]);
        let expand28_out1 = {
            let onnx_shape: [i64; 4usize] = [1, 128, 1, 100];
            let input_dims = unsqueeze29_out1.dims();
            let mut shape = onnx_shape;
            #[allow(clippy::needless_range_loop)]
            for i in 0..4usize {
                let dim_offset = 4usize - 4usize + i;
                if shape[dim_offset] == 1 && input_dims[i] > 1 {
                    shape[dim_offset] = input_dims[i] as i64;
                }
            }
            unsqueeze29_out1.expand(shape)
        };
        let reshape29_out1 = expand28_out1.reshape([1, 128, -1]);
        let slice28_out1 = reshape29_out1.slice(s![.., .., 0..100]);
        let add32_out1 = reducemean28_out1.add(slice28_out1);
        let conv1d113_out1 = self.conv1d113.forward(add32_out1);
        let relu96_out1 = burn::tensor::activation::relu(conv1d113_out1);
        let conv1d114_out1 = self.conv1d114.forward(relu96_out1);
        let sigmoid28_out1 = burn::tensor::activation::sigmoid(conv1d114_out1);
        let mul28_out1 = conv1d112_out1.mul(sigmoid28_out1);
        let concat28_out1 = burn::tensor::Tensor::cat(
            [concat27_out1, mul28_out1].into(),
            1,
        );
        let batchnormalization30_out1 = self
            .batchnormalization30
            .forward(concat28_out1.clone());
        let relu97_out1 = burn::tensor::activation::relu(batchnormalization30_out1);
        let conv1d115_out1 = self.conv1d115.forward(relu97_out1);
        let relu98_out1 = burn::tensor::activation::relu(conv1d115_out1);
        let conv1d116_out1 = self.conv1d116.forward(relu98_out1.clone());
        let reducemean29_out1 = { relu98_out1.clone().mean_dim(2usize) };
        let averagepool1d29_out1 = self.averagepool1d29.forward(relu98_out1);
        let unsqueeze30_out1: Tensor<B, 4> = averagepool1d29_out1
            .unsqueeze_dims::<4>(&[-1]);
        let expand29_out1 = {
            let onnx_shape: [i64; 4usize] = [1, 128, 1, 100];
            let input_dims = unsqueeze30_out1.dims();
            let mut shape = onnx_shape;
            #[allow(clippy::needless_range_loop)]
            for i in 0..4usize {
                let dim_offset = 4usize - 4usize + i;
                if shape[dim_offset] == 1 && input_dims[i] > 1 {
                    shape[dim_offset] = input_dims[i] as i64;
                }
            }
            unsqueeze30_out1.expand(shape)
        };
        let reshape30_out1 = expand29_out1.reshape([1, 128, -1]);
        let slice29_out1 = reshape30_out1.slice(s![.., .., 0..100]);
        let add33_out1 = reducemean29_out1.add(slice29_out1);
        let conv1d117_out1 = self.conv1d117.forward(add33_out1);
        let relu99_out1 = burn::tensor::activation::relu(conv1d117_out1);
        let conv1d118_out1 = self.conv1d118.forward(relu99_out1);
        let sigmoid29_out1 = burn::tensor::activation::sigmoid(conv1d118_out1);
        let mul29_out1 = conv1d116_out1.mul(sigmoid29_out1);
        let concat29_out1 = burn::tensor::Tensor::cat(
            [concat28_out1, mul29_out1].into(),
            1,
        );
        let batchnormalization31_out1 = self
            .batchnormalization31
            .forward(concat29_out1.clone());
        let relu100_out1 = burn::tensor::activation::relu(batchnormalization31_out1);
        let conv1d119_out1 = self.conv1d119.forward(relu100_out1);
        let relu101_out1 = burn::tensor::activation::relu(conv1d119_out1);
        let conv1d120_out1 = self.conv1d120.forward(relu101_out1.clone());
        let reducemean30_out1 = { relu101_out1.clone().mean_dim(2usize) };
        let averagepool1d30_out1 = self.averagepool1d30.forward(relu101_out1);
        let unsqueeze31_out1: Tensor<B, 4> = averagepool1d30_out1
            .unsqueeze_dims::<4>(&[-1]);
        let expand30_out1 = {
            let onnx_shape: [i64; 4usize] = [1, 128, 1, 100];
            let input_dims = unsqueeze31_out1.dims();
            let mut shape = onnx_shape;
            #[allow(clippy::needless_range_loop)]
            for i in 0..4usize {
                let dim_offset = 4usize - 4usize + i;
                if shape[dim_offset] == 1 && input_dims[i] > 1 {
                    shape[dim_offset] = input_dims[i] as i64;
                }
            }
            unsqueeze31_out1.expand(shape)
        };
        let reshape31_out1 = expand30_out1.reshape([1, 128, -1]);
        let slice30_out1 = reshape31_out1.slice(s![.., .., 0..100]);
        let add34_out1 = reducemean30_out1.add(slice30_out1);
        let conv1d121_out1 = self.conv1d121.forward(add34_out1);
        let relu102_out1 = burn::tensor::activation::relu(conv1d121_out1);
        let conv1d122_out1 = self.conv1d122.forward(relu102_out1);
        let sigmoid30_out1 = burn::tensor::activation::sigmoid(conv1d122_out1);
        let mul30_out1 = conv1d120_out1.mul(sigmoid30_out1);
        let concat30_out1 = burn::tensor::Tensor::cat(
            [concat29_out1, mul30_out1].into(),
            1,
        );
        let batchnormalization32_out1 = self
            .batchnormalization32
            .forward(concat30_out1.clone());
        let relu103_out1 = burn::tensor::activation::relu(batchnormalization32_out1);
        let conv1d123_out1 = self.conv1d123.forward(relu103_out1);
        let relu104_out1 = burn::tensor::activation::relu(conv1d123_out1);
        let conv1d124_out1 = self.conv1d124.forward(relu104_out1.clone());
        let reducemean31_out1 = { relu104_out1.clone().mean_dim(2usize) };
        let averagepool1d31_out1 = self.averagepool1d31.forward(relu104_out1);
        let unsqueeze32_out1: Tensor<B, 4> = averagepool1d31_out1
            .unsqueeze_dims::<4>(&[-1]);
        let expand31_out1 = {
            let onnx_shape: [i64; 4usize] = [1, 128, 1, 100];
            let input_dims = unsqueeze32_out1.dims();
            let mut shape = onnx_shape;
            #[allow(clippy::needless_range_loop)]
            for i in 0..4usize {
                let dim_offset = 4usize - 4usize + i;
                if shape[dim_offset] == 1 && input_dims[i] > 1 {
                    shape[dim_offset] = input_dims[i] as i64;
                }
            }
            unsqueeze32_out1.expand(shape)
        };
        let reshape32_out1 = expand31_out1.reshape([1, 128, -1]);
        let slice31_out1 = reshape32_out1.slice(s![.., .., 0..100]);
        let add35_out1 = reducemean31_out1.add(slice31_out1);
        let conv1d125_out1 = self.conv1d125.forward(add35_out1);
        let relu105_out1 = burn::tensor::activation::relu(conv1d125_out1);
        let conv1d126_out1 = self.conv1d126.forward(relu105_out1);
        let sigmoid31_out1 = burn::tensor::activation::sigmoid(conv1d126_out1);
        let mul31_out1 = conv1d124_out1.mul(sigmoid31_out1);
        let concat31_out1 = burn::tensor::Tensor::cat(
            [concat30_out1, mul31_out1].into(),
            1,
        );
        let batchnormalization33_out1 = self
            .batchnormalization33
            .forward(concat31_out1.clone());
        let relu106_out1 = burn::tensor::activation::relu(batchnormalization33_out1);
        let conv1d127_out1 = self.conv1d127.forward(relu106_out1);
        let relu107_out1 = burn::tensor::activation::relu(conv1d127_out1);
        let conv1d128_out1 = self.conv1d128.forward(relu107_out1.clone());
        let reducemean32_out1 = { relu107_out1.clone().mean_dim(2usize) };
        let averagepool1d32_out1 = self.averagepool1d32.forward(relu107_out1);
        let unsqueeze33_out1: Tensor<B, 4> = averagepool1d32_out1
            .unsqueeze_dims::<4>(&[-1]);
        let expand32_out1 = {
            let onnx_shape: [i64; 4usize] = [1, 128, 1, 100];
            let input_dims = unsqueeze33_out1.dims();
            let mut shape = onnx_shape;
            #[allow(clippy::needless_range_loop)]
            for i in 0..4usize {
                let dim_offset = 4usize - 4usize + i;
                if shape[dim_offset] == 1 && input_dims[i] > 1 {
                    shape[dim_offset] = input_dims[i] as i64;
                }
            }
            unsqueeze33_out1.expand(shape)
        };
        let reshape33_out1 = expand32_out1.reshape([1, 128, -1]);
        let slice32_out1 = reshape33_out1.slice(s![.., .., 0..100]);
        let add36_out1 = reducemean32_out1.add(slice32_out1);
        let conv1d129_out1 = self.conv1d129.forward(add36_out1);
        let relu108_out1 = burn::tensor::activation::relu(conv1d129_out1);
        let conv1d130_out1 = self.conv1d130.forward(relu108_out1);
        let sigmoid32_out1 = burn::tensor::activation::sigmoid(conv1d130_out1);
        let mul32_out1 = conv1d128_out1.mul(sigmoid32_out1);
        let concat32_out1 = burn::tensor::Tensor::cat(
            [concat31_out1, mul32_out1].into(),
            1,
        );
        let batchnormalization34_out1 = self
            .batchnormalization34
            .forward(concat32_out1.clone());
        let relu109_out1 = burn::tensor::activation::relu(batchnormalization34_out1);
        let conv1d131_out1 = self.conv1d131.forward(relu109_out1);
        let relu110_out1 = burn::tensor::activation::relu(conv1d131_out1);
        let conv1d132_out1 = self.conv1d132.forward(relu110_out1.clone());
        let reducemean33_out1 = { relu110_out1.clone().mean_dim(2usize) };
        let averagepool1d33_out1 = self.averagepool1d33.forward(relu110_out1);
        let unsqueeze34_out1: Tensor<B, 4> = averagepool1d33_out1
            .unsqueeze_dims::<4>(&[-1]);
        let expand33_out1 = {
            let onnx_shape: [i64; 4usize] = [1, 128, 1, 100];
            let input_dims = unsqueeze34_out1.dims();
            let mut shape = onnx_shape;
            #[allow(clippy::needless_range_loop)]
            for i in 0..4usize {
                let dim_offset = 4usize - 4usize + i;
                if shape[dim_offset] == 1 && input_dims[i] > 1 {
                    shape[dim_offset] = input_dims[i] as i64;
                }
            }
            unsqueeze34_out1.expand(shape)
        };
        let reshape34_out1 = expand33_out1.reshape([1, 128, -1]);
        let slice33_out1 = reshape34_out1.slice(s![.., .., 0..100]);
        let add37_out1 = reducemean33_out1.add(slice33_out1);
        let conv1d133_out1 = self.conv1d133.forward(add37_out1);
        let relu111_out1 = burn::tensor::activation::relu(conv1d133_out1);
        let conv1d134_out1 = self.conv1d134.forward(relu111_out1);
        let sigmoid33_out1 = burn::tensor::activation::sigmoid(conv1d134_out1);
        let mul33_out1 = conv1d132_out1.mul(sigmoid33_out1);
        let concat33_out1 = burn::tensor::Tensor::cat(
            [concat32_out1, mul33_out1].into(),
            1,
        );
        let batchnormalization35_out1 = self
            .batchnormalization35
            .forward(concat33_out1.clone());
        let relu112_out1 = burn::tensor::activation::relu(batchnormalization35_out1);
        let conv1d135_out1 = self.conv1d135.forward(relu112_out1);
        let relu113_out1 = burn::tensor::activation::relu(conv1d135_out1);
        let conv1d136_out1 = self.conv1d136.forward(relu113_out1.clone());
        let reducemean34_out1 = { relu113_out1.clone().mean_dim(2usize) };
        let averagepool1d34_out1 = self.averagepool1d34.forward(relu113_out1);
        let unsqueeze35_out1: Tensor<B, 4> = averagepool1d34_out1
            .unsqueeze_dims::<4>(&[-1]);
        let expand34_out1 = {
            let onnx_shape: [i64; 4usize] = [1, 128, 1, 100];
            let input_dims = unsqueeze35_out1.dims();
            let mut shape = onnx_shape;
            #[allow(clippy::needless_range_loop)]
            for i in 0..4usize {
                let dim_offset = 4usize - 4usize + i;
                if shape[dim_offset] == 1 && input_dims[i] > 1 {
                    shape[dim_offset] = input_dims[i] as i64;
                }
            }
            unsqueeze35_out1.expand(shape)
        };
        let reshape35_out1 = expand34_out1.reshape([1, 128, -1]);
        let slice34_out1 = reshape35_out1.slice(s![.., .., 0..100]);
        let add38_out1 = reducemean34_out1.add(slice34_out1);
        let conv1d137_out1 = self.conv1d137.forward(add38_out1);
        let relu114_out1 = burn::tensor::activation::relu(conv1d137_out1);
        let conv1d138_out1 = self.conv1d138.forward(relu114_out1);
        let sigmoid34_out1 = burn::tensor::activation::sigmoid(conv1d138_out1);
        let mul34_out1 = conv1d136_out1.mul(sigmoid34_out1);
        let concat34_out1 = burn::tensor::Tensor::cat(
            [concat33_out1, mul34_out1].into(),
            1,
        );
        let batchnormalization36_out1 = self
            .batchnormalization36
            .forward(concat34_out1.clone());
        let relu115_out1 = burn::tensor::activation::relu(batchnormalization36_out1);
        let conv1d139_out1 = self.conv1d139.forward(relu115_out1);
        let relu116_out1 = burn::tensor::activation::relu(conv1d139_out1);
        let conv1d140_out1 = self.conv1d140.forward(relu116_out1.clone());
        let reducemean35_out1 = { relu116_out1.clone().mean_dim(2usize) };
        let averagepool1d35_out1 = self.averagepool1d35.forward(relu116_out1);
        let unsqueeze36_out1: Tensor<B, 4> = averagepool1d35_out1
            .unsqueeze_dims::<4>(&[-1]);
        let expand35_out1 = {
            let onnx_shape: [i64; 4usize] = [1, 128, 1, 100];
            let input_dims = unsqueeze36_out1.dims();
            let mut shape = onnx_shape;
            #[allow(clippy::needless_range_loop)]
            for i in 0..4usize {
                let dim_offset = 4usize - 4usize + i;
                if shape[dim_offset] == 1 && input_dims[i] > 1 {
                    shape[dim_offset] = input_dims[i] as i64;
                }
            }
            unsqueeze36_out1.expand(shape)
        };
        let reshape36_out1 = expand35_out1.reshape([1, 128, -1]);
        let slice35_out1 = reshape36_out1.slice(s![.., .., 0..100]);
        let add39_out1 = reducemean35_out1.add(slice35_out1);
        let conv1d141_out1 = self.conv1d141.forward(add39_out1);
        let relu117_out1 = burn::tensor::activation::relu(conv1d141_out1);
        let conv1d142_out1 = self.conv1d142.forward(relu117_out1);
        let sigmoid35_out1 = burn::tensor::activation::sigmoid(conv1d142_out1);
        let mul35_out1 = conv1d140_out1.mul(sigmoid35_out1);
        let concat35_out1 = burn::tensor::Tensor::cat(
            [concat34_out1, mul35_out1].into(),
            1,
        );
        let batchnormalization37_out1 = self
            .batchnormalization37
            .forward(concat35_out1.clone());
        let relu118_out1 = burn::tensor::activation::relu(batchnormalization37_out1);
        let conv1d143_out1 = self.conv1d143.forward(relu118_out1);
        let relu119_out1 = burn::tensor::activation::relu(conv1d143_out1);
        let conv1d144_out1 = self.conv1d144.forward(relu119_out1.clone());
        let reducemean36_out1 = { relu119_out1.clone().mean_dim(2usize) };
        let averagepool1d36_out1 = self.averagepool1d36.forward(relu119_out1);
        let unsqueeze37_out1: Tensor<B, 4> = averagepool1d36_out1
            .unsqueeze_dims::<4>(&[-1]);
        let expand36_out1 = {
            let onnx_shape: [i64; 4usize] = [1, 128, 1, 100];
            let input_dims = unsqueeze37_out1.dims();
            let mut shape = onnx_shape;
            #[allow(clippy::needless_range_loop)]
            for i in 0..4usize {
                let dim_offset = 4usize - 4usize + i;
                if shape[dim_offset] == 1 && input_dims[i] > 1 {
                    shape[dim_offset] = input_dims[i] as i64;
                }
            }
            unsqueeze37_out1.expand(shape)
        };
        let reshape37_out1 = expand36_out1.reshape([1, 128, -1]);
        let slice36_out1 = reshape37_out1.slice(s![.., .., 0..100]);
        let add40_out1 = reducemean36_out1.add(slice36_out1);
        let conv1d145_out1 = self.conv1d145.forward(add40_out1);
        let relu120_out1 = burn::tensor::activation::relu(conv1d145_out1);
        let conv1d146_out1 = self.conv1d146.forward(relu120_out1);
        let sigmoid36_out1 = burn::tensor::activation::sigmoid(conv1d146_out1);
        let mul36_out1 = conv1d144_out1.mul(sigmoid36_out1);
        let concat36_out1 = burn::tensor::Tensor::cat(
            [concat35_out1, mul36_out1].into(),
            1,
        );
        let batchnormalization38_out1 = self.batchnormalization38.forward(concat36_out1);
        let relu121_out1 = burn::tensor::activation::relu(batchnormalization38_out1);
        let conv1d147_out1 = self.conv1d147.forward(relu121_out1);
        let batchnormalization39_out1 = self
            .batchnormalization39
            .forward(conv1d147_out1.clone());
        let relu122_out1 = burn::tensor::activation::relu(batchnormalization39_out1);
        let conv1d148_out1 = self.conv1d148.forward(relu122_out1);
        let relu123_out1 = burn::tensor::activation::relu(conv1d148_out1);
        let conv1d149_out1 = self.conv1d149.forward(relu123_out1.clone());
        let reducemean37_out1 = { relu123_out1.clone().mean_dim(2usize) };
        let averagepool1d37_out1 = self.averagepool1d37.forward(relu123_out1);
        let unsqueeze38_out1: Tensor<B, 4> = averagepool1d37_out1
            .unsqueeze_dims::<4>(&[-1]);
        let expand37_out1 = {
            let onnx_shape: [i64; 4usize] = [1, 128, 1, 100];
            let input_dims = unsqueeze38_out1.dims();
            let mut shape = onnx_shape;
            #[allow(clippy::needless_range_loop)]
            for i in 0..4usize {
                let dim_offset = 4usize - 4usize + i;
                if shape[dim_offset] == 1 && input_dims[i] > 1 {
                    shape[dim_offset] = input_dims[i] as i64;
                }
            }
            unsqueeze38_out1.expand(shape)
        };
        let reshape38_out1 = expand37_out1.reshape([1, 128, -1]);
        let slice37_out1 = reshape38_out1.slice(s![.., .., 0..100]);
        let add41_out1 = reducemean37_out1.add(slice37_out1);
        let conv1d150_out1 = self.conv1d150.forward(add41_out1);
        let relu124_out1 = burn::tensor::activation::relu(conv1d150_out1);
        let conv1d151_out1 = self.conv1d151.forward(relu124_out1);
        let sigmoid37_out1 = burn::tensor::activation::sigmoid(conv1d151_out1);
        let mul37_out1 = conv1d149_out1.mul(sigmoid37_out1);
        let concat37_out1 = burn::tensor::Tensor::cat(
            [conv1d147_out1, mul37_out1].into(),
            1,
        );
        let batchnormalization40_out1 = self
            .batchnormalization40
            .forward(concat37_out1.clone());
        let relu125_out1 = burn::tensor::activation::relu(batchnormalization40_out1);
        let conv1d152_out1 = self.conv1d152.forward(relu125_out1);
        let relu126_out1 = burn::tensor::activation::relu(conv1d152_out1);
        let conv1d153_out1 = self.conv1d153.forward(relu126_out1.clone());
        let reducemean38_out1 = { relu126_out1.clone().mean_dim(2usize) };
        let averagepool1d38_out1 = self.averagepool1d38.forward(relu126_out1);
        let unsqueeze39_out1: Tensor<B, 4> = averagepool1d38_out1
            .unsqueeze_dims::<4>(&[-1]);
        let expand38_out1 = {
            let onnx_shape: [i64; 4usize] = [1, 128, 1, 100];
            let input_dims = unsqueeze39_out1.dims();
            let mut shape = onnx_shape;
            #[allow(clippy::needless_range_loop)]
            for i in 0..4usize {
                let dim_offset = 4usize - 4usize + i;
                if shape[dim_offset] == 1 && input_dims[i] > 1 {
                    shape[dim_offset] = input_dims[i] as i64;
                }
            }
            unsqueeze39_out1.expand(shape)
        };
        let reshape39_out1 = expand38_out1.reshape([1, 128, -1]);
        let slice38_out1 = reshape39_out1.slice(s![.., .., 0..100]);
        let add42_out1 = reducemean38_out1.add(slice38_out1);
        let conv1d154_out1 = self.conv1d154.forward(add42_out1);
        let relu127_out1 = burn::tensor::activation::relu(conv1d154_out1);
        let conv1d155_out1 = self.conv1d155.forward(relu127_out1);
        let sigmoid38_out1 = burn::tensor::activation::sigmoid(conv1d155_out1);
        let mul38_out1 = conv1d153_out1.mul(sigmoid38_out1);
        let concat38_out1 = burn::tensor::Tensor::cat(
            [concat37_out1, mul38_out1].into(),
            1,
        );
        let batchnormalization41_out1 = self
            .batchnormalization41
            .forward(concat38_out1.clone());
        let relu128_out1 = burn::tensor::activation::relu(batchnormalization41_out1);
        let conv1d156_out1 = self.conv1d156.forward(relu128_out1);
        let relu129_out1 = burn::tensor::activation::relu(conv1d156_out1);
        let conv1d157_out1 = self.conv1d157.forward(relu129_out1.clone());
        let reducemean39_out1 = { relu129_out1.clone().mean_dim(2usize) };
        let averagepool1d39_out1 = self.averagepool1d39.forward(relu129_out1);
        let unsqueeze40_out1: Tensor<B, 4> = averagepool1d39_out1
            .unsqueeze_dims::<4>(&[-1]);
        let expand39_out1 = {
            let onnx_shape: [i64; 4usize] = [1, 128, 1, 100];
            let input_dims = unsqueeze40_out1.dims();
            let mut shape = onnx_shape;
            #[allow(clippy::needless_range_loop)]
            for i in 0..4usize {
                let dim_offset = 4usize - 4usize + i;
                if shape[dim_offset] == 1 && input_dims[i] > 1 {
                    shape[dim_offset] = input_dims[i] as i64;
                }
            }
            unsqueeze40_out1.expand(shape)
        };
        let reshape40_out1 = expand39_out1.reshape([1, 128, -1]);
        let slice39_out1 = reshape40_out1.slice(s![.., .., 0..100]);
        let add43_out1 = reducemean39_out1.add(slice39_out1);
        let conv1d158_out1 = self.conv1d158.forward(add43_out1);
        let relu130_out1 = burn::tensor::activation::relu(conv1d158_out1);
        let conv1d159_out1 = self.conv1d159.forward(relu130_out1);
        let sigmoid39_out1 = burn::tensor::activation::sigmoid(conv1d159_out1);
        let mul39_out1 = conv1d157_out1.mul(sigmoid39_out1);
        let concat39_out1 = burn::tensor::Tensor::cat(
            [concat38_out1, mul39_out1].into(),
            1,
        );
        let batchnormalization42_out1 = self
            .batchnormalization42
            .forward(concat39_out1.clone());
        let relu131_out1 = burn::tensor::activation::relu(batchnormalization42_out1);
        let conv1d160_out1 = self.conv1d160.forward(relu131_out1);
        let relu132_out1 = burn::tensor::activation::relu(conv1d160_out1);
        let conv1d161_out1 = self.conv1d161.forward(relu132_out1.clone());
        let reducemean40_out1 = { relu132_out1.clone().mean_dim(2usize) };
        let averagepool1d40_out1 = self.averagepool1d40.forward(relu132_out1);
        let unsqueeze41_out1: Tensor<B, 4> = averagepool1d40_out1
            .unsqueeze_dims::<4>(&[-1]);
        let expand40_out1 = {
            let onnx_shape: [i64; 4usize] = [1, 128, 1, 100];
            let input_dims = unsqueeze41_out1.dims();
            let mut shape = onnx_shape;
            #[allow(clippy::needless_range_loop)]
            for i in 0..4usize {
                let dim_offset = 4usize - 4usize + i;
                if shape[dim_offset] == 1 && input_dims[i] > 1 {
                    shape[dim_offset] = input_dims[i] as i64;
                }
            }
            unsqueeze41_out1.expand(shape)
        };
        let reshape41_out1 = expand40_out1.reshape([1, 128, -1]);
        let slice40_out1 = reshape41_out1.slice(s![.., .., 0..100]);
        let add44_out1 = reducemean40_out1.add(slice40_out1);
        let conv1d162_out1 = self.conv1d162.forward(add44_out1);
        let relu133_out1 = burn::tensor::activation::relu(conv1d162_out1);
        let conv1d163_out1 = self.conv1d163.forward(relu133_out1);
        let sigmoid40_out1 = burn::tensor::activation::sigmoid(conv1d163_out1);
        let mul40_out1 = conv1d161_out1.mul(sigmoid40_out1);
        let concat40_out1 = burn::tensor::Tensor::cat(
            [concat39_out1, mul40_out1].into(),
            1,
        );
        let batchnormalization43_out1 = self
            .batchnormalization43
            .forward(concat40_out1.clone());
        let relu134_out1 = burn::tensor::activation::relu(batchnormalization43_out1);
        let conv1d164_out1 = self.conv1d164.forward(relu134_out1);
        let relu135_out1 = burn::tensor::activation::relu(conv1d164_out1);
        let conv1d165_out1 = self.conv1d165.forward(relu135_out1.clone());
        let reducemean41_out1 = { relu135_out1.clone().mean_dim(2usize) };
        let averagepool1d41_out1 = self.averagepool1d41.forward(relu135_out1);
        let unsqueeze42_out1: Tensor<B, 4> = averagepool1d41_out1
            .unsqueeze_dims::<4>(&[-1]);
        let expand41_out1 = {
            let onnx_shape: [i64; 4usize] = [1, 128, 1, 100];
            let input_dims = unsqueeze42_out1.dims();
            let mut shape = onnx_shape;
            #[allow(clippy::needless_range_loop)]
            for i in 0..4usize {
                let dim_offset = 4usize - 4usize + i;
                if shape[dim_offset] == 1 && input_dims[i] > 1 {
                    shape[dim_offset] = input_dims[i] as i64;
                }
            }
            unsqueeze42_out1.expand(shape)
        };
        let reshape42_out1 = expand41_out1.reshape([1, 128, -1]);
        let slice41_out1 = reshape42_out1.slice(s![.., .., 0..100]);
        let add45_out1 = reducemean41_out1.add(slice41_out1);
        let conv1d166_out1 = self.conv1d166.forward(add45_out1);
        let relu136_out1 = burn::tensor::activation::relu(conv1d166_out1);
        let conv1d167_out1 = self.conv1d167.forward(relu136_out1);
        let sigmoid41_out1 = burn::tensor::activation::sigmoid(conv1d167_out1);
        let mul41_out1 = conv1d165_out1.mul(sigmoid41_out1);
        let concat41_out1 = burn::tensor::Tensor::cat(
            [concat40_out1, mul41_out1].into(),
            1,
        );
        let batchnormalization44_out1 = self
            .batchnormalization44
            .forward(concat41_out1.clone());
        let relu137_out1 = burn::tensor::activation::relu(batchnormalization44_out1);
        let conv1d168_out1 = self.conv1d168.forward(relu137_out1);
        let relu138_out1 = burn::tensor::activation::relu(conv1d168_out1);
        let conv1d169_out1 = self.conv1d169.forward(relu138_out1.clone());
        let reducemean42_out1 = { relu138_out1.clone().mean_dim(2usize) };
        let averagepool1d42_out1 = self.averagepool1d42.forward(relu138_out1);
        let unsqueeze43_out1: Tensor<B, 4> = averagepool1d42_out1
            .unsqueeze_dims::<4>(&[-1]);
        let expand42_out1 = {
            let onnx_shape: [i64; 4usize] = [1, 128, 1, 100];
            let input_dims = unsqueeze43_out1.dims();
            let mut shape = onnx_shape;
            #[allow(clippy::needless_range_loop)]
            for i in 0..4usize {
                let dim_offset = 4usize - 4usize + i;
                if shape[dim_offset] == 1 && input_dims[i] > 1 {
                    shape[dim_offset] = input_dims[i] as i64;
                }
            }
            unsqueeze43_out1.expand(shape)
        };
        let reshape43_out1 = expand42_out1.reshape([1, 128, -1]);
        let slice42_out1 = reshape43_out1.slice(s![.., .., 0..100]);
        let add46_out1 = reducemean42_out1.add(slice42_out1);
        let conv1d170_out1 = self.conv1d170.forward(add46_out1);
        let relu139_out1 = burn::tensor::activation::relu(conv1d170_out1);
        let conv1d171_out1 = self.conv1d171.forward(relu139_out1);
        let sigmoid42_out1 = burn::tensor::activation::sigmoid(conv1d171_out1);
        let mul42_out1 = conv1d169_out1.mul(sigmoid42_out1);
        let concat42_out1 = burn::tensor::Tensor::cat(
            [concat41_out1, mul42_out1].into(),
            1,
        );
        let batchnormalization45_out1 = self
            .batchnormalization45
            .forward(concat42_out1.clone());
        let relu140_out1 = burn::tensor::activation::relu(batchnormalization45_out1);
        let conv1d172_out1 = self.conv1d172.forward(relu140_out1);
        let relu141_out1 = burn::tensor::activation::relu(conv1d172_out1);
        let conv1d173_out1 = self.conv1d173.forward(relu141_out1.clone());
        let reducemean43_out1 = { relu141_out1.clone().mean_dim(2usize) };
        let averagepool1d43_out1 = self.averagepool1d43.forward(relu141_out1);
        let unsqueeze44_out1: Tensor<B, 4> = averagepool1d43_out1
            .unsqueeze_dims::<4>(&[-1]);
        let expand43_out1 = {
            let onnx_shape: [i64; 4usize] = [1, 128, 1, 100];
            let input_dims = unsqueeze44_out1.dims();
            let mut shape = onnx_shape;
            #[allow(clippy::needless_range_loop)]
            for i in 0..4usize {
                let dim_offset = 4usize - 4usize + i;
                if shape[dim_offset] == 1 && input_dims[i] > 1 {
                    shape[dim_offset] = input_dims[i] as i64;
                }
            }
            unsqueeze44_out1.expand(shape)
        };
        let reshape44_out1 = expand43_out1.reshape([1, 128, -1]);
        let slice43_out1 = reshape44_out1.slice(s![.., .., 0..100]);
        let add47_out1 = reducemean43_out1.add(slice43_out1);
        let conv1d174_out1 = self.conv1d174.forward(add47_out1);
        let relu142_out1 = burn::tensor::activation::relu(conv1d174_out1);
        let conv1d175_out1 = self.conv1d175.forward(relu142_out1);
        let sigmoid43_out1 = burn::tensor::activation::sigmoid(conv1d175_out1);
        let mul43_out1 = conv1d173_out1.mul(sigmoid43_out1);
        let concat43_out1 = burn::tensor::Tensor::cat(
            [concat42_out1, mul43_out1].into(),
            1,
        );
        let batchnormalization46_out1 = self
            .batchnormalization46
            .forward(concat43_out1.clone());
        let relu143_out1 = burn::tensor::activation::relu(batchnormalization46_out1);
        let conv1d176_out1 = self.conv1d176.forward(relu143_out1);
        let relu144_out1 = burn::tensor::activation::relu(conv1d176_out1);
        let conv1d177_out1 = self.conv1d177.forward(relu144_out1.clone());
        let reducemean44_out1 = { relu144_out1.clone().mean_dim(2usize) };
        let averagepool1d44_out1 = self.averagepool1d44.forward(relu144_out1);
        let unsqueeze45_out1: Tensor<B, 4> = averagepool1d44_out1
            .unsqueeze_dims::<4>(&[-1]);
        let expand44_out1 = {
            let onnx_shape: [i64; 4usize] = [1, 128, 1, 100];
            let input_dims = unsqueeze45_out1.dims();
            let mut shape = onnx_shape;
            #[allow(clippy::needless_range_loop)]
            for i in 0..4usize {
                let dim_offset = 4usize - 4usize + i;
                if shape[dim_offset] == 1 && input_dims[i] > 1 {
                    shape[dim_offset] = input_dims[i] as i64;
                }
            }
            unsqueeze45_out1.expand(shape)
        };
        let reshape45_out1 = expand44_out1.reshape([1, 128, -1]);
        let slice44_out1 = reshape45_out1.slice(s![.., .., 0..100]);
        let add48_out1 = reducemean44_out1.add(slice44_out1);
        let conv1d178_out1 = self.conv1d178.forward(add48_out1);
        let relu145_out1 = burn::tensor::activation::relu(conv1d178_out1);
        let conv1d179_out1 = self.conv1d179.forward(relu145_out1);
        let sigmoid44_out1 = burn::tensor::activation::sigmoid(conv1d179_out1);
        let mul44_out1 = conv1d177_out1.mul(sigmoid44_out1);
        let concat44_out1 = burn::tensor::Tensor::cat(
            [concat43_out1, mul44_out1].into(),
            1,
        );
        let batchnormalization47_out1 = self
            .batchnormalization47
            .forward(concat44_out1.clone());
        let relu146_out1 = burn::tensor::activation::relu(batchnormalization47_out1);
        let conv1d180_out1 = self.conv1d180.forward(relu146_out1);
        let relu147_out1 = burn::tensor::activation::relu(conv1d180_out1);
        let conv1d181_out1 = self.conv1d181.forward(relu147_out1.clone());
        let reducemean45_out1 = { relu147_out1.clone().mean_dim(2usize) };
        let averagepool1d45_out1 = self.averagepool1d45.forward(relu147_out1);
        let unsqueeze46_out1: Tensor<B, 4> = averagepool1d45_out1
            .unsqueeze_dims::<4>(&[-1]);
        let expand45_out1 = {
            let onnx_shape: [i64; 4usize] = [1, 128, 1, 100];
            let input_dims = unsqueeze46_out1.dims();
            let mut shape = onnx_shape;
            #[allow(clippy::needless_range_loop)]
            for i in 0..4usize {
                let dim_offset = 4usize - 4usize + i;
                if shape[dim_offset] == 1 && input_dims[i] > 1 {
                    shape[dim_offset] = input_dims[i] as i64;
                }
            }
            unsqueeze46_out1.expand(shape)
        };
        let reshape46_out1 = expand45_out1.reshape([1, 128, -1]);
        let slice45_out1 = reshape46_out1.slice(s![.., .., 0..100]);
        let add49_out1 = reducemean45_out1.add(slice45_out1);
        let conv1d182_out1 = self.conv1d182.forward(add49_out1);
        let relu148_out1 = burn::tensor::activation::relu(conv1d182_out1);
        let conv1d183_out1 = self.conv1d183.forward(relu148_out1);
        let sigmoid45_out1 = burn::tensor::activation::sigmoid(conv1d183_out1);
        let mul45_out1 = conv1d181_out1.mul(sigmoid45_out1);
        let concat45_out1 = burn::tensor::Tensor::cat(
            [concat44_out1, mul45_out1].into(),
            1,
        );
        let batchnormalization48_out1 = self
            .batchnormalization48
            .forward(concat45_out1.clone());
        let relu149_out1 = burn::tensor::activation::relu(batchnormalization48_out1);
        let conv1d184_out1 = self.conv1d184.forward(relu149_out1);
        let relu150_out1 = burn::tensor::activation::relu(conv1d184_out1);
        let conv1d185_out1 = self.conv1d185.forward(relu150_out1.clone());
        let reducemean46_out1 = { relu150_out1.clone().mean_dim(2usize) };
        let averagepool1d46_out1 = self.averagepool1d46.forward(relu150_out1);
        let unsqueeze47_out1: Tensor<B, 4> = averagepool1d46_out1
            .unsqueeze_dims::<4>(&[-1]);
        let expand46_out1 = {
            let onnx_shape: [i64; 4usize] = [1, 128, 1, 100];
            let input_dims = unsqueeze47_out1.dims();
            let mut shape = onnx_shape;
            #[allow(clippy::needless_range_loop)]
            for i in 0..4usize {
                let dim_offset = 4usize - 4usize + i;
                if shape[dim_offset] == 1 && input_dims[i] > 1 {
                    shape[dim_offset] = input_dims[i] as i64;
                }
            }
            unsqueeze47_out1.expand(shape)
        };
        let reshape47_out1 = expand46_out1.reshape([1, 128, -1]);
        let slice46_out1 = reshape47_out1.slice(s![.., .., 0..100]);
        let add50_out1 = reducemean46_out1.add(slice46_out1);
        let conv1d186_out1 = self.conv1d186.forward(add50_out1);
        let relu151_out1 = burn::tensor::activation::relu(conv1d186_out1);
        let conv1d187_out1 = self.conv1d187.forward(relu151_out1);
        let sigmoid46_out1 = burn::tensor::activation::sigmoid(conv1d187_out1);
        let mul46_out1 = conv1d185_out1.mul(sigmoid46_out1);
        let concat46_out1 = burn::tensor::Tensor::cat(
            [concat45_out1, mul46_out1].into(),
            1,
        );
        let batchnormalization49_out1 = self
            .batchnormalization49
            .forward(concat46_out1.clone());
        let relu152_out1 = burn::tensor::activation::relu(batchnormalization49_out1);
        let conv1d188_out1 = self.conv1d188.forward(relu152_out1);
        let relu153_out1 = burn::tensor::activation::relu(conv1d188_out1);
        let conv1d189_out1 = self.conv1d189.forward(relu153_out1.clone());
        let reducemean47_out1 = { relu153_out1.clone().mean_dim(2usize) };
        let averagepool1d47_out1 = self.averagepool1d47.forward(relu153_out1);
        let unsqueeze48_out1: Tensor<B, 4> = averagepool1d47_out1
            .unsqueeze_dims::<4>(&[-1]);
        let expand47_out1 = {
            let onnx_shape: [i64; 4usize] = [1, 128, 1, 100];
            let input_dims = unsqueeze48_out1.dims();
            let mut shape = onnx_shape;
            #[allow(clippy::needless_range_loop)]
            for i in 0..4usize {
                let dim_offset = 4usize - 4usize + i;
                if shape[dim_offset] == 1 && input_dims[i] > 1 {
                    shape[dim_offset] = input_dims[i] as i64;
                }
            }
            unsqueeze48_out1.expand(shape)
        };
        let reshape48_out1 = expand47_out1.reshape([1, 128, -1]);
        let slice47_out1 = reshape48_out1.slice(s![.., .., 0..100]);
        let add51_out1 = reducemean47_out1.add(slice47_out1);
        let conv1d190_out1 = self.conv1d190.forward(add51_out1);
        let relu154_out1 = burn::tensor::activation::relu(conv1d190_out1);
        let conv1d191_out1 = self.conv1d191.forward(relu154_out1);
        let sigmoid47_out1 = burn::tensor::activation::sigmoid(conv1d191_out1);
        let mul47_out1 = conv1d189_out1.mul(sigmoid47_out1);
        let concat47_out1 = burn::tensor::Tensor::cat(
            [concat46_out1, mul47_out1].into(),
            1,
        );
        let batchnormalization50_out1 = self
            .batchnormalization50
            .forward(concat47_out1.clone());
        let relu155_out1 = burn::tensor::activation::relu(batchnormalization50_out1);
        let conv1d192_out1 = self.conv1d192.forward(relu155_out1);
        let relu156_out1 = burn::tensor::activation::relu(conv1d192_out1);
        let conv1d193_out1 = self.conv1d193.forward(relu156_out1.clone());
        let reducemean48_out1 = { relu156_out1.clone().mean_dim(2usize) };
        let averagepool1d48_out1 = self.averagepool1d48.forward(relu156_out1);
        let unsqueeze49_out1: Tensor<B, 4> = averagepool1d48_out1
            .unsqueeze_dims::<4>(&[-1]);
        let expand48_out1 = {
            let onnx_shape: [i64; 4usize] = [1, 128, 1, 100];
            let input_dims = unsqueeze49_out1.dims();
            let mut shape = onnx_shape;
            #[allow(clippy::needless_range_loop)]
            for i in 0..4usize {
                let dim_offset = 4usize - 4usize + i;
                if shape[dim_offset] == 1 && input_dims[i] > 1 {
                    shape[dim_offset] = input_dims[i] as i64;
                }
            }
            unsqueeze49_out1.expand(shape)
        };
        let reshape49_out1 = expand48_out1.reshape([1, 128, -1]);
        let slice48_out1 = reshape49_out1.slice(s![.., .., 0..100]);
        let add52_out1 = reducemean48_out1.add(slice48_out1);
        let conv1d194_out1 = self.conv1d194.forward(add52_out1);
        let relu157_out1 = burn::tensor::activation::relu(conv1d194_out1);
        let conv1d195_out1 = self.conv1d195.forward(relu157_out1);
        let sigmoid48_out1 = burn::tensor::activation::sigmoid(conv1d195_out1);
        let mul48_out1 = conv1d193_out1.mul(sigmoid48_out1);
        let concat48_out1 = burn::tensor::Tensor::cat(
            [concat47_out1, mul48_out1].into(),
            1,
        );
        let batchnormalization51_out1 = self
            .batchnormalization51
            .forward(concat48_out1.clone());
        let relu158_out1 = burn::tensor::activation::relu(batchnormalization51_out1);
        let conv1d196_out1 = self.conv1d196.forward(relu158_out1);
        let relu159_out1 = burn::tensor::activation::relu(conv1d196_out1);
        let conv1d197_out1 = self.conv1d197.forward(relu159_out1.clone());
        let reducemean49_out1 = { relu159_out1.clone().mean_dim(2usize) };
        let averagepool1d49_out1 = self.averagepool1d49.forward(relu159_out1);
        let unsqueeze50_out1: Tensor<B, 4> = averagepool1d49_out1
            .unsqueeze_dims::<4>(&[-1]);
        let expand49_out1 = {
            let onnx_shape: [i64; 4usize] = [1, 128, 1, 100];
            let input_dims = unsqueeze50_out1.dims();
            let mut shape = onnx_shape;
            #[allow(clippy::needless_range_loop)]
            for i in 0..4usize {
                let dim_offset = 4usize - 4usize + i;
                if shape[dim_offset] == 1 && input_dims[i] > 1 {
                    shape[dim_offset] = input_dims[i] as i64;
                }
            }
            unsqueeze50_out1.expand(shape)
        };
        let reshape50_out1 = expand49_out1.reshape([1, 128, -1]);
        let slice49_out1 = reshape50_out1.slice(s![.., .., 0..100]);
        let add53_out1 = reducemean49_out1.add(slice49_out1);
        let conv1d198_out1 = self.conv1d198.forward(add53_out1);
        let relu160_out1 = burn::tensor::activation::relu(conv1d198_out1);
        let conv1d199_out1 = self.conv1d199.forward(relu160_out1);
        let sigmoid49_out1 = burn::tensor::activation::sigmoid(conv1d199_out1);
        let mul49_out1 = conv1d197_out1.mul(sigmoid49_out1);
        let concat49_out1 = burn::tensor::Tensor::cat(
            [concat48_out1, mul49_out1].into(),
            1,
        );
        let batchnormalization52_out1 = self
            .batchnormalization52
            .forward(concat49_out1.clone());
        let relu161_out1 = burn::tensor::activation::relu(batchnormalization52_out1);
        let conv1d200_out1 = self.conv1d200.forward(relu161_out1);
        let relu162_out1 = burn::tensor::activation::relu(conv1d200_out1);
        let conv1d201_out1 = self.conv1d201.forward(relu162_out1.clone());
        let reducemean50_out1 = { relu162_out1.clone().mean_dim(2usize) };
        let averagepool1d50_out1 = self.averagepool1d50.forward(relu162_out1);
        let unsqueeze51_out1: Tensor<B, 4> = averagepool1d50_out1
            .unsqueeze_dims::<4>(&[-1]);
        let expand50_out1 = {
            let onnx_shape: [i64; 4usize] = [1, 128, 1, 100];
            let input_dims = unsqueeze51_out1.dims();
            let mut shape = onnx_shape;
            #[allow(clippy::needless_range_loop)]
            for i in 0..4usize {
                let dim_offset = 4usize - 4usize + i;
                if shape[dim_offset] == 1 && input_dims[i] > 1 {
                    shape[dim_offset] = input_dims[i] as i64;
                }
            }
            unsqueeze51_out1.expand(shape)
        };
        let reshape51_out1 = expand50_out1.reshape([1, 128, -1]);
        let slice50_out1 = reshape51_out1.slice(s![.., .., 0..100]);
        let add54_out1 = reducemean50_out1.add(slice50_out1);
        let conv1d202_out1 = self.conv1d202.forward(add54_out1);
        let relu163_out1 = burn::tensor::activation::relu(conv1d202_out1);
        let conv1d203_out1 = self.conv1d203.forward(relu163_out1);
        let sigmoid50_out1 = burn::tensor::activation::sigmoid(conv1d203_out1);
        let mul50_out1 = conv1d201_out1.mul(sigmoid50_out1);
        let concat50_out1 = burn::tensor::Tensor::cat(
            [concat49_out1, mul50_out1].into(),
            1,
        );
        let batchnormalization53_out1 = self
            .batchnormalization53
            .forward(concat50_out1.clone());
        let relu164_out1 = burn::tensor::activation::relu(batchnormalization53_out1);
        let conv1d204_out1 = self.conv1d204.forward(relu164_out1);
        let relu165_out1 = burn::tensor::activation::relu(conv1d204_out1);
        let conv1d205_out1 = self.conv1d205.forward(relu165_out1.clone());
        let reducemean51_out1 = { relu165_out1.clone().mean_dim(2usize) };
        let averagepool1d51_out1 = self.averagepool1d51.forward(relu165_out1);
        let unsqueeze52_out1: Tensor<B, 4> = averagepool1d51_out1
            .unsqueeze_dims::<4>(&[-1]);
        let expand51_out1 = {
            let onnx_shape: [i64; 4usize] = [1, 128, 1, 100];
            let input_dims = unsqueeze52_out1.dims();
            let mut shape = onnx_shape;
            #[allow(clippy::needless_range_loop)]
            for i in 0..4usize {
                let dim_offset = 4usize - 4usize + i;
                if shape[dim_offset] == 1 && input_dims[i] > 1 {
                    shape[dim_offset] = input_dims[i] as i64;
                }
            }
            unsqueeze52_out1.expand(shape)
        };
        let reshape52_out1 = expand51_out1.reshape([1, 128, -1]);
        let slice51_out1 = reshape52_out1.slice(s![.., .., 0..100]);
        let add55_out1 = reducemean51_out1.add(slice51_out1);
        let conv1d206_out1 = self.conv1d206.forward(add55_out1);
        let relu166_out1 = burn::tensor::activation::relu(conv1d206_out1);
        let conv1d207_out1 = self.conv1d207.forward(relu166_out1);
        let sigmoid51_out1 = burn::tensor::activation::sigmoid(conv1d207_out1);
        let mul51_out1 = conv1d205_out1.mul(sigmoid51_out1);
        let concat51_out1 = burn::tensor::Tensor::cat(
            [concat50_out1, mul51_out1].into(),
            1,
        );
        let batchnormalization54_out1 = self
            .batchnormalization54
            .forward(concat51_out1.clone());
        let relu167_out1 = burn::tensor::activation::relu(batchnormalization54_out1);
        let conv1d208_out1 = self.conv1d208.forward(relu167_out1);
        let relu168_out1 = burn::tensor::activation::relu(conv1d208_out1);
        let conv1d209_out1 = self.conv1d209.forward(relu168_out1.clone());
        let reducemean52_out1 = { relu168_out1.clone().mean_dim(2usize) };
        let averagepool1d52_out1 = self.averagepool1d52.forward(relu168_out1);
        let unsqueeze53_out1: Tensor<B, 4> = averagepool1d52_out1
            .unsqueeze_dims::<4>(&[-1]);
        let expand52_out1 = {
            let onnx_shape: [i64; 4usize] = [1, 128, 1, 100];
            let input_dims = unsqueeze53_out1.dims();
            let mut shape = onnx_shape;
            #[allow(clippy::needless_range_loop)]
            for i in 0..4usize {
                let dim_offset = 4usize - 4usize + i;
                if shape[dim_offset] == 1 && input_dims[i] > 1 {
                    shape[dim_offset] = input_dims[i] as i64;
                }
            }
            unsqueeze53_out1.expand(shape)
        };
        let reshape53_out1 = expand52_out1.reshape([1, 128, -1]);
        let slice52_out1 = reshape53_out1.slice(s![.., .., 0..100]);
        let add56_out1 = reducemean52_out1.add(slice52_out1);
        let conv1d210_out1 = self.conv1d210.forward(add56_out1);
        let relu169_out1 = burn::tensor::activation::relu(conv1d210_out1);
        let conv1d211_out1 = self.conv1d211.forward(relu169_out1);
        let sigmoid52_out1 = burn::tensor::activation::sigmoid(conv1d211_out1);
        let mul52_out1 = conv1d209_out1.mul(sigmoid52_out1);
        let concat52_out1 = burn::tensor::Tensor::cat(
            [concat51_out1, mul52_out1].into(),
            1,
        );
        let batchnormalization55_out1 = self.batchnormalization55.forward(concat52_out1);
        let relu170_out1 = burn::tensor::activation::relu(batchnormalization55_out1);
        let conv1d212_out1 = self.conv1d212.forward(relu170_out1);
        let relu171_out1 = burn::tensor::activation::relu(conv1d212_out1);
        let reducemean53_out1 = {
            relu171_out1.clone().mean_dim(2usize).squeeze_dims::<2usize>(&[2])
        };
        let reducemean54_out1 = { relu171_out1.clone().mean_dim(2usize) };
        let sub1_out1 = relu171_out1.sub(reducemean54_out1);
        let mul53_out1 = sub1_out1.clone().mul(sub1_out1);
        let reducemean55_out1 = {
            mul53_out1.mean_dim(2usize).squeeze_dims::<2usize>(&[2])
        };
        let mul54_out1 = reducemean55_out1.mul_scalar(constant629_out1);
        let div1_out1 = mul54_out1.div_scalar(constant630_out1);
        let add57_out1 = div1_out1.add_scalar(constant622_out1);
        let sqrt1_out1 = add57_out1.sqrt();
        let concat53_out1 = burn::tensor::Tensor::cat(
            [reducemean53_out1, sqrt1_out1].into(),
            1,
        );
        let unsqueeze54_out1: Tensor<B, 3> = concat53_out1.unsqueeze_dims::<3>(&[-1]);
        let conv1d213_out1 = self.conv1d213.forward(unsqueeze54_out1);
        let squeeze1_out1 = conv1d213_out1.squeeze_dims::<2>(&[2]);
        let batchnormalization56_out1 = self.batchnormalization56.forward(squeeze1_out1);
        batchnormalization56_out1
    }
}
