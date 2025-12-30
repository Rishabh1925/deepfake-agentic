import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Upload, 
  Video, 
  Scissors, 
  Eye, 
  Volume2, 
  Brain, 
  Zap, 
  Target,
  CheckCircle,
  AlertCircle,
  Play,
  Pause,
  RotateCcw,
  ArrowRight,
  Database,
  Cpu,
  Network
} from 'lucide-react';

const DataPipeline = () => {
  const [currentStage, setCurrentStage] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [dataPackets, setDataPackets] = useState<Array<{id: number, stage: number, progress: number}>>([]);

  const stages = [
    {
      id: 'upload',
      title: 'Video Upload',
      description: 'User uploads video file through the web interface',
      icon: Upload,
      color: 'from-blue-500 to-cyan-500',
      processes: [
        'File validation',
        'Format checking',
        'Size verification',
        'Security scan'
      ],
      dataFlow: {
        input: 'Video File (.mp4, .avi, .mov)',
        output: 'Validated Video Stream',
        size: '~50MB',
        format: 'H.264/AVC'
      }
    },
    {
      id: 'preprocessing',
      title: 'Preprocessing Pipeline',
      description: 'Parallel extraction of frames and audio components',
      icon: Scissors,
      color: 'from-purple-500 to-pink-500',
      processes: [
        'Video decoding',
        'Frame extraction (30 FPS)',
        'Audio extraction',
        'Metadata parsing'
      ],
      dataFlow: {
        input: 'Video Stream',
        output: 'Frames + Audio + Metadata',
        size: '~150 frames + 3MB audio',
        format: 'RGB frames + WAV'
      }
    },
    {
      id: 'face_detection',
      title: 'Face Detection',
      description: 'MTCNN-based face detection and cropping',
      icon: Eye,
      color: 'from-green-500 to-emerald-500',
      processes: [
        'MTCNN face detection',
        'Face alignment',
        'Crop extraction',
        'Quality filtering'
      ],
      dataFlow: {
        input: 'Video Frames [224x224x3]',
        output: 'Face Crops [224x224x3]',
        size: '~100 face crops',
        format: 'Normalized RGB'
      }
    },
    {
      id: 'audio_processing',
      title: 'Audio Processing',
      description: 'Mel spectrogram generation for audio analysis',
      icon: Volume2,
      color: 'from-orange-500 to-red-500',
      processes: [
        'Audio normalization',
        'STFT computation',
        'Mel filterbank',
        'Spectrogram generation'
      ],
      dataFlow: {
        input: 'Audio WAV [16kHz]',
        output: 'Mel Spectrogram [128xT]',
        size: '~128x300 matrix',
        format: 'Float32 tensor'
      }
    },
    {
      id: 'baseline_inference',
      title: 'Baseline Model',
      description: 'BG-Model processes input for initial prediction',
      icon: Cpu,
      color: 'from-indigo-500 to-purple-500',
      processes: [
        'MobileNetV3 feature extraction',
        'Temporal convolution',
        'Global average pooling',
        'Classification head'
      ],
      dataFlow: {
        input: 'Face Crops [Bx3x224x224]',
        output: 'Prediction + Confidence',
        size: '2 logits + confidence score',
        format: '[fake_prob, real_prob]'
      }
    },
    {
      id: 'routing_decision',
      title: 'Intelligent Routing',
      description: 'Agent evaluates confidence and routes to specialists',
      icon: Brain,
      color: 'from-pink-500 to-rose-500',
      processes: [
        'Confidence evaluation',
        'Domain classification',
        'Specialist selection',
        'Resource allocation'
      ],
      dataFlow: {
        input: 'Baseline Prediction + Metadata',
        output: 'Routing Decision',
        size: 'Boolean flags per specialist',
        format: '{av: true, cm: false, ...}'
      }
    },
    {
      id: 'specialist_inference',
      title: 'Specialist Models',
      description: 'Domain-specific models provide specialized analysis',
      icon: Network,
      color: 'from-teal-500 to-cyan-500',
      processes: [
        'AV-Model (lip-sync)',
        'CM-Model (compression)',
        'LL-Model (low-light)',
        'TM-Model (temporal)'
      ],
      dataFlow: {
        input: 'Preprocessed Data',
        output: 'Specialist Predictions',
        size: 'Multiple prediction vectors',
        format: 'Weighted confidence scores'
      }
    },
    {
      id: 'aggregation',
      title: 'Result Aggregation',
      description: 'Ensemble combines predictions with bias correction',
      icon: Target,
      color: 'from-yellow-500 to-orange-500',
      processes: [
        'Weighted ensemble',
        'Confidence calibration',
        'Bias correction',
        'Final prediction'
      ],
      dataFlow: {
        input: 'All Model Predictions',
        output: 'Final Result + Explanation',
        size: 'Prediction + confidence + reasoning',
        format: 'JSON response'
      }
    }
  ];

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isPlaying) {
      interval = setInterval(() => {
        setCurrentStage((prev) => (prev + 1) % stages.length);
        
        // Add new data packet
        setDataPackets(prev => [
          ...prev,
          { id: Date.now(), stage: 0, progress: 0 }
        ]);
      }, 3000);
    }
    return () => clearInterval(interval);
  }, [isPlaying, stages.length]);

  // Animate data packets
  useEffect(() => {
    const interval = setInterval(() => {
      setDataPackets(prev => 
        prev.map(packet => ({
          ...packet,
          progress: packet.progress + 2,
          stage: Math.floor(packet.progress / 12.5)
        })).filter(packet => packet.progress < 100)
      );
    }, 100);
    return () => clearInterval(interval);
  }, []);

  const handlePlayPause = () => {
    setIsPlaying(!isPlaying);
  };

  const handleReset = () => {
    setCurrentStage(0);
    setDataPackets([]);
    setIsPlaying(false);
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
            Data Pipeline Flow
          </h1>
          <p className="text-xl text-gray-300 max-w-4xl mx-auto mb-8">
            Follow the complete journey of data from video upload to final deepfake detection result. 
            Watch how information flows through each processing stage with real-time visualization.
          </p>
          
          {/* Controls */}
          <div className="flex items-center justify-center gap-4">
            <button
              onClick={handlePlayPause}
              className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-cyan-500 to-purple-500 rounded-xl text-white font-medium hover:scale-105 transition-transform"
            >
              {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
              {isPlaying ? 'Pause' : 'Start'} Pipeline
            </button>
            <button
              onClick={handleReset}
              className="flex items-center gap-2 px-6 py-3 bg-gray-700 rounded-xl text-white font-medium hover:bg-gray-600 transition-colors"
            >
              <RotateCcw className="w-5 h-5" />
              Reset
            </button>
          </div>
        </motion.div>

        {/* Pipeline Visualization */}
        <div className="relative mb-16">
          {/* Pipeline Stages */}
          <div className="flex flex-wrap justify-center gap-4 mb-8">
            {stages.map((stage, index) => {
              const Icon = stage.icon;
              const isActive = currentStage >= index;
              const isCurrent = currentStage === index;
              
              return (
                <motion.div
                  key={stage.id}
                  className={`relative ${isCurrent ? 'z-20' : 'z-10'}`}
                  initial={{ scale: 0, opacity: 0 }}
                  animate={{ 
                    scale: isActive ? 1 : 0.8,
                    opacity: isActive ? 1 : 0.4
                  }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                >
                  {/* Pulse Effect for Current Stage */}
                  {isCurrent && (
                    <motion.div
                      className={`absolute inset-0 rounded-2xl bg-gradient-to-r ${stage.color} opacity-30`}
                      animate={{ scale: [1, 1.2, 1] }}
                      transition={{ duration: 2, repeat: Infinity }}
                    />
                  )}
                  
                  {/* Stage Card */}
                  <div className={`w-48 bg-black/20 backdrop-blur-sm rounded-2xl p-4 border-2 ${
                    isCurrent ? 'border-white/40' : 'border-white/10'
                  } hover:border-white/20 transition-all duration-300`}>
                    <div className={`w-12 h-12 rounded-xl bg-gradient-to-r ${stage.color} flex items-center justify-center mb-3`}>
                      <Icon className="w-6 h-6 text-white" />
                    </div>
                    <h3 className="text-white font-semibold mb-2">{stage.title}</h3>
                    <p className="text-gray-300 text-sm">{stage.description}</p>
                    
                    {/* Progress Indicator */}
                    {isCurrent && (
                      <div className="mt-3">
                        <div className="w-full bg-gray-700 rounded-full h-2">
                          <motion.div
                            className={`h-2 rounded-full bg-gradient-to-r ${stage.color}`}
                            initial={{ width: 0 }}
                            animate={{ width: '100%' }}
                            transition={{ duration: 2.5, repeat: Infinity }}
                          />
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Data Flow Indicator */}
                  {isCurrent && (
                    <motion.div
                      className="absolute -top-2 -right-2 w-6 h-6 bg-yellow-400 rounded-full flex items-center justify-center"
                      animate={{ scale: [1, 1.3, 1] }}
                      transition={{ duration: 0.8, repeat: Infinity }}
                    >
                      <Zap className="w-3 h-3 text-black" />
                    </motion.div>
                  )}
                </motion.div>
              );
            })}
          </div>

          {/* Data Flow Animation */}
          <div className="relative h-32 bg-black/10 rounded-2xl border border-white/10 overflow-hidden mb-8">
            <div className="absolute inset-0 flex items-center">
              {/* Pipeline Track */}
              <div className="w-full h-2 bg-gradient-to-r from-cyan-500 via-purple-500 to-pink-500 opacity-30 rounded-full"></div>
              
              {/* Animated Data Packets */}
              {dataPackets.map((packet) => (
                <motion.div
                  key={packet.id}
                  className="absolute w-4 h-4 bg-gradient-to-r from-cyan-400 to-purple-400 rounded-full shadow-lg"
                  style={{ left: `${packet.progress}%` }}
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  exit={{ scale: 0 }}
                >
                  <div className="absolute inset-0 bg-white/30 rounded-full animate-ping"></div>
                </motion.div>
              ))}
            </div>
            
            {/* Stage Markers */}
            {stages.map((_, index) => (
              <div
                key={index}
                className="absolute top-1/2 transform -translate-y-1/2 w-3 h-3 bg-white rounded-full border-2 border-gray-600"
                style={{ left: `${(index / (stages.length - 1)) * 100}%` }}
              />
            ))}
          </div>
        </div>

        {/* Current Stage Details */}
        <AnimatePresence mode="wait">
          {currentStage < stages.length && (
            <motion.div
              key={currentStage}
              initial={{ opacity: 0, y: 40 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -40 }}
              className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-16"
            >
              {/* Stage Info */}
              <div className="bg-black/20 backdrop-blur-sm rounded-3xl border border-white/10 p-8">
                <div className="flex items-center gap-4 mb-6">
                  <div className={`w-16 h-16 rounded-2xl bg-gradient-to-r ${stages[currentStage].color} flex items-center justify-center`}>
                    {React.createElement(stages[currentStage].icon, { className: "w-8 h-8 text-white" })}
                  </div>
                  <div>
                    <h3 className="text-2xl font-bold text-white mb-2">{stages[currentStage].title}</h3>
                    <p className="text-gray-300">{stages[currentStage].description}</p>
                  </div>
                </div>
                
                <h4 className="text-lg font-semibold text-white mb-4">Processing Steps:</h4>
                <div className="space-y-3">
                  {stages[currentStage].processes.map((process, index) => (
                    <motion.div
                      key={index}
                      className="flex items-center gap-3"
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.2 }}
                    >
                      <CheckCircle className="w-5 h-5 text-green-400" />
                      <span className="text-gray-300">{process}</span>
                    </motion.div>
                  ))}
                </div>
              </div>

              {/* Data Flow Info */}
              <div className="bg-black/20 backdrop-blur-sm rounded-3xl border border-white/10 p-8">
                <h3 className="text-2xl font-bold text-white mb-6">Data Flow</h3>
                
                <div className="space-y-6">
                  <div>
                    <div className="flex items-center gap-2 mb-2">
                      <ArrowRight className="w-4 h-4 text-cyan-400" />
                      <span className="text-sm font-semibold text-cyan-400">INPUT</span>
                    </div>
                    <div className="bg-black/30 rounded-xl p-4">
                      <div className="text-white font-mono">{stages[currentStage].dataFlow.input}</div>
                      <div className="text-gray-400 text-sm mt-1">Size: {stages[currentStage].dataFlow.size}</div>
                    </div>
                  </div>
                  
                  <div className="flex justify-center">
                    <motion.div
                      className="w-12 h-12 rounded-full bg-gradient-to-r from-purple-500 to-pink-500 flex items-center justify-center"
                      animate={{ rotate: 360 }}
                      transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                    >
                      <Cpu className="w-6 h-6 text-white" />
                    </motion.div>
                  </div>
                  
                  <div>
                    <div className="flex items-center gap-2 mb-2">
                      <ArrowRight className="w-4 h-4 text-green-400" />
                      <span className="text-sm font-semibold text-green-400">OUTPUT</span>
                    </div>
                    <div className="bg-black/30 rounded-xl p-4">
                      <div className="text-white font-mono">{stages[currentStage].dataFlow.output}</div>
                      <div className="text-gray-400 text-sm mt-1">Format: {stages[currentStage].dataFlow.format}</div>
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Pipeline Statistics */}
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="bg-black/20 backdrop-blur-sm rounded-3xl border border-white/10 p-8"
        >
          <h3 className="text-2xl font-bold text-white mb-8 text-center">Pipeline Performance</h3>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            <div className="text-center">
              <div className="w-16 h-16 rounded-2xl bg-gradient-to-r from-blue-500 to-cyan-500 flex items-center justify-center mx-auto mb-3">
                <Upload className="w-8 h-8 text-white" />
              </div>
              <div className="text-2xl font-bold text-cyan-400 mb-1">~2s</div>
              <div className="text-gray-400 text-sm">Upload & Validation</div>
            </div>
            
            <div className="text-center">
              <div className="w-16 h-16 rounded-2xl bg-gradient-to-r from-purple-500 to-pink-500 flex items-center justify-center mx-auto mb-3">
                <Scissors className="w-8 h-8 text-white" />
              </div>
              <div className="text-2xl font-bold text-purple-400 mb-1">~3s</div>
              <div className="text-gray-400 text-sm">Preprocessing</div>
            </div>
            
            <div className="text-center">
              <div className="w-16 h-16 rounded-2xl bg-gradient-to-r from-green-500 to-emerald-500 flex items-center justify-center mx-auto mb-3">
                <Brain className="w-8 h-8 text-white" />
              </div>
              <div className="text-2xl font-bold text-green-400 mb-1">~1s</div>
              <div className="text-gray-400 text-sm">Model Inference</div>
            </div>
            
            <div className="text-center">
              <div className="w-16 h-16 rounded-2xl bg-gradient-to-r from-orange-500 to-red-500 flex items-center justify-center mx-auto mb-3">
                <Target className="w-8 h-8 text-white" />
              </div>
              <div className="text-2xl font-bold text-orange-400 mb-1">~0.5s</div>
              <div className="text-gray-400 text-sm">Result Aggregation</div>
            </div>
          </div>
          
          <div className="mt-8 text-center">
            <div className="text-4xl font-bold bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent mb-2">
              ~6.5s
            </div>
            <div className="text-gray-300">Total Pipeline Latency</div>
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default DataPipeline;