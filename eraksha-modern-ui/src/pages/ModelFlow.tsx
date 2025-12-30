import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Cpu, 
  Layers, 
  Zap, 
  Eye, 
  Volume2, 
  Image, 
  Clock, 
  Compress,
  Monitor,
  Sun,
  ChevronDown,
  ChevronUp,
  Play,
  Pause
} from 'lucide-react';

const ModelFlow = () => {
  const [selectedModel, setSelectedModel] = useState('bg');
  const [isAnimating, setIsAnimating] = useState(false);
  const [layerIndex, setLayerIndex] = useState(0);

  const models = {
    bg: {
      name: 'BG-Model (Baseline)',
      description: 'MobileNetV3-based generalist model for initial deepfake detection',
      params: '2.1M',
      accuracy: '86.25%',
      latency: '45ms',
      color: 'from-blue-500 to-cyan-500',
      icon: Cpu,
      layers: [
        { name: 'Input', type: 'input', shape: '[B, T, 3, 224, 224]', description: 'Video frames batch' },
        { name: 'Conv Stem', type: 'conv', shape: '[B, 16, 112, 112]', description: 'Initial convolution' },
        { name: 'MobileNet Blocks', type: 'mobilenet', shape: '[B, 96, 7, 7]', description: 'Inverted residual blocks' },
        { name: 'Temporal Conv', type: 'conv', shape: '[B, 128, 1, 1]', description: 'Temporal aggregation' },
        { name: 'Global Pool', type: 'pool', shape: '[B, 128]', description: 'Spatial pooling' },
        { name: 'Classifier', type: 'fc', shape: '[B, 2]', description: 'Final prediction' }
      ]
    },
    av: {
      name: 'AV-Model (Audio-Visual)',
      description: 'Multi-modal model analyzing both visual and audio features for lip-sync detection',
      params: '15.8M',
      accuracy: '93.0%',
      latency: '120ms',
      color: 'from-purple-500 to-pink-500',
      icon: Volume2,
      layers: [
        { name: 'Video Input', type: 'input', shape: '[B, T, 3, 224, 224]', description: 'Video frames' },
        { name: 'Audio Input', type: 'input', shape: '[B, 1, 128, T]', description: 'Mel spectrogram' },
        { name: 'ResNet18 (Visual)', type: 'resnet', shape: '[B, 512]', description: 'Visual features' },
        { name: 'Audio CNN', type: 'conv', shape: '[B, 128]', description: 'Audio features' },
        { name: 'Cross Attention', type: 'attention', shape: '[B, 512]', description: 'Multi-modal fusion' },
        { name: 'Lip-Sync Analyzer', type: 'fc', shape: '[B, 1]', description: 'Synchronization score' },
        { name: 'Final Classifier', type: 'fc', shape: '[B, 2]', description: 'Deepfake prediction' }
      ]
    },
    cm: {
      name: 'CM-Model (Compression)',
      description: 'Specialist model for detecting compression artifacts in deepfake videos',
      params: '11.7M',
      accuracy: '80.83%',
      latency: '85ms',
      color: 'from-orange-500 to-red-500',
      icon: Compress,
      layers: [
        { name: 'Input', type: 'input', shape: '[B, 3, 224, 224]', description: 'Compressed frames' },
        { name: 'ResNet18 Backbone', type: 'resnet', shape: '[B, 512, 7, 7]', description: 'Feature extraction' },
        { name: 'Artifact Detector', type: 'conv', shape: '[B, 256, 7, 7]', description: 'Compression artifacts' },
        { name: 'Frequency Analysis', type: 'fft', shape: '[B, 128]', description: 'DCT coefficients' },
        { name: 'Feature Fusion', type: 'concat', shape: '[B, 640]', description: 'Combined features' },
        { name: 'Classifier', type: 'fc', shape: '[B, 2]', description: 'Compression detection' }
      ]
    },
    ll: {
      name: 'LL-Model (Low-Light)',
      description: 'Specialist model optimized for low-light and poor quality video detection',
      params: '11.7M',
      accuracy: '93.42%',
      latency: '85ms',
      color: 'from-green-500 to-emerald-500',
      icon: Sun,
      layers: [
        { name: 'Input', type: 'input', shape: '[B, 3, 224, 224]', description: 'Low-light frames' },
        { name: 'Enhancement Layer', type: 'enhance', shape: '[B, 3, 224, 224]', description: 'Brightness adjustment' },
        { name: 'ResNet18 Backbone', type: 'resnet', shape: '[B, 512, 7, 7]', description: 'Robust features' },
        { name: 'Noise Detector', type: 'conv', shape: '[B, 128, 7, 7]', description: 'Noise patterns' },
        { name: 'Quality Estimator', type: 'fc', shape: '[B, 64]', description: 'Image quality' },
        { name: 'Adaptive Classifier', type: 'fc', shape: '[B, 2]', description: 'Quality-aware prediction' }
      ]
    },
    tm: {
      name: 'TM-Model (Temporal)',
      description: 'LSTM-based model for analyzing temporal inconsistencies across video frames',
      params: '14.2M',
      accuracy: '78.5%',
      latency: '150ms',
      color: 'from-indigo-500 to-purple-500',
      icon: Clock,
      layers: [
        { name: 'Input Sequence', type: 'input', shape: '[B, T, 3, 224, 224]', description: 'Video sequence' },
        { name: 'Per-Frame ResNet', type: 'resnet', shape: '[B, T, 512]', description: 'Frame features' },
        { name: 'LSTM Layer 1', type: 'lstm', shape: '[B, T, 256]', description: 'Temporal modeling' },
        { name: 'LSTM Layer 2', type: 'lstm', shape: '[B, T, 256]', description: 'Deep temporal features' },
        { name: 'Temporal Attention', type: 'attention', shape: '[B, 256]', description: 'Important frames' },
        { name: 'Sequence Classifier', type: 'fc', shape: '[B, 2]', description: 'Temporal prediction' }
      ]
    }
  };

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isAnimating) {
      interval = setInterval(() => {
        setLayerIndex((prev) => (prev + 1) % models[selectedModel as keyof typeof models].layers.length);
      }, 1500);
    }
    return () => clearInterval(interval);
  }, [isAnimating, selectedModel]);

  const currentModel = models[selectedModel as keyof typeof models];
  const Icon = currentModel.icon;

  const getLayerColor = (type: string) => {
    const colors = {
      input: 'from-gray-500 to-gray-600',
      conv: 'from-blue-500 to-blue-600',
      resnet: 'from-purple-500 to-purple-600',
      mobilenet: 'from-cyan-500 to-cyan-600',
      lstm: 'from-green-500 to-green-600',
      attention: 'from-pink-500 to-pink-600',
      fc: 'from-orange-500 to-orange-600',
      pool: 'from-red-500 to-red-600',
      enhance: 'from-yellow-500 to-yellow-600',
      fft: 'from-indigo-500 to-indigo-600',
      concat: 'from-teal-500 to-teal-600'
    };
    return colors[type as keyof typeof colors] || 'from-gray-500 to-gray-600';
  };

  return (
    <div className="min-h-screen pt-32 pb-20 px-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-16"
        >
          <h1 className="text-6xl font-bold bg-gradient-to-r from-cyan-400 via-purple-400 to-pink-400 bg-clip-text text-transparent mb-6">
            Neural Network Architecture Flow
          </h1>
          <p className="text-xl text-gray-300 max-w-4xl mx-auto mb-8">
            Explore the detailed architecture of each specialist model in the E-Raksha system. 
            Watch data flow through layers and understand how each model processes information.
          </p>
        </motion.div>

        {/* Model Selector */}
        <div className="flex flex-wrap justify-center gap-4 mb-12">
          {Object.entries(models).map(([key, model]) => {
            const ModelIcon = model.icon;
            return (
              <button
                key={key}
                onClick={() => {
                  setSelectedModel(key);
                  setLayerIndex(0);
                }}
                className={`flex items-center gap-3 px-6 py-4 rounded-2xl border-2 transition-all duration-300 ${
                  selectedModel === key
                    ? 'border-white/30 bg-white/10 scale-105'
                    : 'border-white/10 bg-black/20 hover:border-white/20 hover:bg-white/5'
                }`}
              >
                <div className={`w-10 h-10 rounded-lg bg-gradient-to-r ${model.color} flex items-center justify-center`}>
                  <ModelIcon className="w-5 h-5 text-white" />
                </div>
                <div className="text-left">
                  <div className="text-white font-semibold">{model.name}</div>
                  <div className="text-gray-400 text-sm">{model.params} • {model.accuracy}</div>
                </div>
              </button>
            );
          })}
        </div>

        {/* Model Details */}
        <motion.div
          key={selectedModel}
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="bg-black/20 backdrop-blur-sm rounded-3xl border border-white/10 p-8 mb-12"
        >
          <div className="flex items-center justify-between mb-8">
            <div className="flex items-center gap-4">
              <div className={`w-16 h-16 rounded-2xl bg-gradient-to-r ${currentModel.color} flex items-center justify-center`}>
                <Icon className="w-8 h-8 text-white" />
              </div>
              <div>
                <h2 className="text-3xl font-bold text-white mb-2">{currentModel.name}</h2>
                <p className="text-gray-300 max-w-2xl">{currentModel.description}</p>
              </div>
            </div>
            <div className="flex items-center gap-4">
              <button
                onClick={() => setIsAnimating(!isAnimating)}
                className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-cyan-500 to-purple-500 rounded-xl text-white font-medium hover:scale-105 transition-transform"
              >
                {isAnimating ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
                {isAnimating ? 'Pause' : 'Animate'}
              </button>
            </div>
          </div>

          {/* Model Stats */}
          <div className="grid grid-cols-3 gap-6 mb-8">
            <div className="bg-black/30 rounded-2xl p-4 text-center">
              <div className="text-2xl font-bold text-cyan-400 mb-1">{currentModel.params}</div>
              <div className="text-gray-400">Parameters</div>
            </div>
            <div className="bg-black/30 rounded-2xl p-4 text-center">
              <div className="text-2xl font-bold text-green-400 mb-1">{currentModel.accuracy}</div>
              <div className="text-gray-400">Accuracy</div>
            </div>
            <div className="bg-black/30 rounded-2xl p-4 text-center">
              <div className="text-2xl font-bold text-purple-400 mb-1">{currentModel.latency}</div>
              <div className="text-gray-400">Latency</div>
            </div>
          </div>

          {/* Architecture Flow */}
          <div className="relative">
            <h3 className="text-xl font-bold text-white mb-6">Architecture Flow</h3>
            
            {/* Layer Connections */}
            <div className="flex items-center justify-between mb-8 overflow-x-auto pb-4">
              {currentModel.layers.map((layer, index) => (
                <div key={index} className="flex items-center">
                  <motion.div
                    className={`relative min-w-[200px] ${index === layerIndex ? 'z-20' : 'z-10'}`}
                    initial={{ scale: 0.8, opacity: 0.6 }}
                    animate={{ 
                      scale: index === layerIndex ? 1.1 : 0.9,
                      opacity: index <= layerIndex ? 1 : 0.4
                    }}
                    transition={{ duration: 0.5 }}
                  >
                    {/* Pulse Effect for Current Layer */}
                    {index === layerIndex && (
                      <motion.div
                        className={`absolute inset-0 rounded-2xl bg-gradient-to-r ${getLayerColor(layer.type)} opacity-30`}
                        animate={{ scale: [1, 1.1, 1] }}
                        transition={{ duration: 1.5, repeat: Infinity }}
                      />
                    )}
                    
                    {/* Layer Block */}
                    <div className={`bg-gradient-to-r ${getLayerColor(layer.type)} rounded-2xl p-4 border-2 ${
                      index === layerIndex ? 'border-white/40' : 'border-white/10'
                    }`}>
                      <div className="text-white font-semibold mb-1">{layer.name}</div>
                      <div className="text-white/80 text-sm mb-2">{layer.shape}</div>
                      <div className="text-white/60 text-xs">{layer.description}</div>
                    </div>

                    {/* Data Flow Indicator */}
                    {index === layerIndex && (
                      <motion.div
                        className="absolute -top-2 -right-2 w-6 h-6 bg-yellow-400 rounded-full flex items-center justify-center"
                        animate={{ scale: [1, 1.3, 1] }}
                        transition={{ duration: 0.8, repeat: Infinity }}
                      >
                        <Zap className="w-3 h-3 text-black" />
                      </motion.div>
                    )}
                  </motion.div>

                  {/* Arrow */}
                  {index < currentModel.layers.length - 1 && (
                    <motion.div
                      className="mx-4 flex items-center"
                      initial={{ opacity: 0.3 }}
                      animate={{ opacity: index < layerIndex ? 1 : 0.3 }}
                    >
                      <div className="w-8 h-0.5 bg-gradient-to-r from-cyan-400 to-purple-400"></div>
                      <div className="w-0 h-0 border-l-4 border-l-purple-400 border-t-2 border-b-2 border-t-transparent border-b-transparent"></div>
                    </motion.div>
                  )}
                </div>
              ))}
            </div>

            {/* Current Layer Details */}
            <AnimatePresence mode="wait">
              <motion.div
                key={layerIndex}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="bg-black/40 backdrop-blur-sm rounded-2xl p-6 border border-white/10"
              >
                <div className="flex items-center gap-4 mb-4">
                  <div className={`w-12 h-12 rounded-xl bg-gradient-to-r ${getLayerColor(currentModel.layers[layerIndex].type)} flex items-center justify-center`}>
                    <Layers className="w-6 h-6 text-white" />
                  </div>
                  <div>
                    <h4 className="text-xl font-bold text-white">{currentModel.layers[layerIndex].name}</h4>
                    <p className="text-gray-300">{currentModel.layers[layerIndex].description}</p>
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <div className="text-sm text-gray-400 mb-1">Output Shape</div>
                    <div className="text-lg font-mono text-cyan-400">{currentModel.layers[layerIndex].shape}</div>
                  </div>
                  <div>
                    <div className="text-sm text-gray-400 mb-1">Layer Type</div>
                    <div className="text-lg font-semibold text-purple-400 capitalize">{currentModel.layers[layerIndex].type}</div>
                  </div>
                </div>
              </motion.div>
            </AnimatePresence>
          </div>
        </motion.div>

        {/* Model Comparison */}
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="bg-black/20 backdrop-blur-sm rounded-3xl border border-white/10 p-8"
        >
          <h3 className="text-2xl font-bold text-white mb-6">Model Comparison</h3>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-white/10">
                  <th className="text-left py-4 px-4 text-gray-300">Model</th>
                  <th className="text-left py-4 px-4 text-gray-300">Specialization</th>
                  <th className="text-left py-4 px-4 text-gray-300">Parameters</th>
                  <th className="text-left py-4 px-4 text-gray-300">Accuracy</th>
                  <th className="text-left py-4 px-4 text-gray-300">Latency</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(models).map(([key, model]) => (
                  <tr key={key} className="border-b border-white/5 hover:bg-white/5 transition-colors">
                    <td className="py-4 px-4">
                      <div className="flex items-center gap-3">
                        <div className={`w-8 h-8 rounded-lg bg-gradient-to-r ${model.color} flex items-center justify-center`}>
                          {React.createElement(model.icon, { className: "w-4 h-4 text-white" })}
                        </div>
                        <span className="text-white font-medium">{model.name}</span>
                      </div>
                    </td>
                    <td className="py-4 px-4 text-gray-300">{model.description.split(' ').slice(0, 6).join(' ')}...</td>
                    <td className="py-4 px-4 text-cyan-400 font-mono">{model.params}</td>
                    <td className="py-4 px-4 text-green-400 font-mono">{model.accuracy}</td>
                    <td className="py-4 px-4 text-purple-400 font-mono">{model.latency}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default ModelFlow;