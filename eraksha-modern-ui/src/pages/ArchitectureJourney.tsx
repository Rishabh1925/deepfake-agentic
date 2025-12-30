import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Upload, 
  Video, 
  Cpu, 
  Network, 
  Brain, 
  Eye, 
  Zap, 
  Shield,
  ChevronRight,
  Play,
  Pause,
  RotateCcw
} from 'lucide-react';

const ArchitectureJourney = () => {
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [dataFlow, setDataFlow] = useState(0);

  const steps = [
    {
      id: 'upload',
      title: 'Video Upload',
      description: 'User uploads a video file for deepfake detection',
      icon: Upload,
      color: 'from-blue-500 to-cyan-500',
      position: { x: 10, y: 50 }
    },
    {
      id: 'preprocessing',
      title: 'Input Processing Layer',
      description: 'Video decoder, frame sampler, face detector, and audio extractor work in parallel',
      icon: Video,
      color: 'from-purple-500 to-pink-500',
      position: { x: 25, y: 30 }
    },
    {
      id: 'baseline',
      title: 'BG-Model Inference',
      description: 'MobileNetV3-based baseline model processes the input',
      icon: Cpu,
      color: 'from-green-500 to-emerald-500',
      position: { x: 40, y: 60 }
    },
    {
      id: 'routing',
      title: 'Intelligent Routing',
      description: 'Agent evaluates confidence and decides on specialist models',
      icon: Brain,
      color: 'from-orange-500 to-red-500',
      position: { x: 55, y: 40 }
    },
    {
      id: 'specialists',
      title: 'Specialist Models',
      description: 'Domain-specific models (AV, CM, RR, LL, TM) provide specialized analysis',
      icon: Network,
      color: 'from-indigo-500 to-purple-500',
      position: { x: 70, y: 20 }
    },
    {
      id: 'aggregation',
      title: 'Result Aggregation',
      description: 'Weighted ensemble combines predictions with bias correction',
      icon: Eye,
      color: 'from-pink-500 to-rose-500',
      position: { x: 85, y: 50 }
    }
  ];

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isPlaying) {
      interval = setInterval(() => {
        setCurrentStep((prev) => (prev + 1) % steps.length);
        setDataFlow((prev) => (prev + 1) % 100);
      }, 2000);
    }
    return () => clearInterval(interval);
  }, [isPlaying, steps.length]);

  const handlePlayPause = () => {
    setIsPlaying(!isPlaying);
  };

  const handleReset = () => {
    setCurrentStep(0);
    setDataFlow(0);
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
            E-Raksha Architecture Journey
          </h1>
          <p className="text-xl text-gray-300 max-w-3xl mx-auto mb-8">
            Experience the complete data flow from video upload to deepfake detection through our 
            four-layer neural architecture system with intelligent agentic routing.
          </p>
          
          {/* Controls */}
          <div className="flex items-center justify-center gap-4">
            <button
              onClick={handlePlayPause}
              className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-cyan-500 to-purple-500 rounded-xl text-white font-medium hover:scale-105 transition-transform"
            >
              {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
              {isPlaying ? 'Pause' : 'Play'} Journey
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

        {/* Architecture Visualization */}
        <div className="relative h-[600px] bg-black/20 rounded-3xl border border-white/10 overflow-hidden mb-16">
          {/* Background Grid */}
          <div className="absolute inset-0 opacity-20">
            <svg className="w-full h-full">
              <defs>
                <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
                  <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#374151" strokeWidth="1"/>
                </pattern>
              </defs>
              <rect width="100%" height="100%" fill="url(#grid)" />
            </svg>
          </div>

          {/* Data Flow Lines */}
          <svg className="absolute inset-0 w-full h-full">
            <defs>
              <linearGradient id="flowGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#06b6d4" />
                <stop offset="50%" stopColor="#8b5cf6" />
                <stop offset="100%" stopColor="#ec4899" />
              </linearGradient>
            </defs>
            {steps.slice(0, -1).map((step, index) => {
              const nextStep = steps[index + 1];
              const isActive = currentStep >= index;
              return (
                <motion.line
                  key={`line-${index}`}
                  x1={`${step.position.x}%`}
                  y1={`${step.position.y}%`}
                  x2={`${nextStep.position.x}%`}
                  y2={`${nextStep.position.y}%`}
                  stroke="url(#flowGradient)"
                  strokeWidth="3"
                  strokeDasharray="10,5"
                  initial={{ pathLength: 0, opacity: 0.3 }}
                  animate={{ 
                    pathLength: isActive ? 1 : 0,
                    opacity: isActive ? 1 : 0.3
                  }}
                  transition={{ duration: 1, ease: "easeInOut" }}
                />
              );
            })}
          </svg>

          {/* Architecture Nodes */}
          {steps.map((step, index) => {
            const Icon = step.icon;
            const isActive = currentStep >= index;
            const isCurrent = currentStep === index;
            
            return (
              <motion.div
                key={step.id}
                className="absolute transform -translate-x-1/2 -translate-y-1/2"
                style={{
                  left: `${step.position.x}%`,
                  top: `${step.position.y}%`
                }}
                initial={{ scale: 0, opacity: 0 }}
                animate={{ 
                  scale: isActive ? 1 : 0.7,
                  opacity: isActive ? 1 : 0.5
                }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
              >
                <div className={`relative ${isCurrent ? 'z-20' : 'z-10'}`}>
                  {/* Pulse Effect for Current Step */}
                  {isCurrent && (
                    <motion.div
                      className={`absolute inset-0 rounded-full bg-gradient-to-r ${step.color} opacity-30`}
                      animate={{ scale: [1, 1.5, 1] }}
                      transition={{ duration: 2, repeat: Infinity }}
                    />
                  )}
                  
                  {/* Node */}
                  <div className={`w-20 h-20 rounded-full bg-gradient-to-r ${step.color} flex items-center justify-center shadow-2xl border-2 border-white/20`}>
                    <Icon className="w-8 h-8 text-white" />
                  </div>
                  
                  {/* Data Flow Indicator */}
                  {isCurrent && (
                    <motion.div
                      className="absolute -top-2 -right-2 w-6 h-6 bg-yellow-400 rounded-full flex items-center justify-center"
                      animate={{ scale: [1, 1.2, 1] }}
                      transition={{ duration: 0.5, repeat: Infinity }}
                    >
                      <Zap className="w-3 h-3 text-black" />
                    </motion.div>
                  )}
                </div>
                
                {/* Label */}
                <div className="absolute top-24 left-1/2 transform -translate-x-1/2 text-center">
                  <div className="bg-black/60 backdrop-blur-sm rounded-lg px-3 py-2 border border-white/10">
                    <h3 className="text-sm font-semibold text-white mb-1">{step.title}</h3>
                    <p className="text-xs text-gray-300 max-w-32">{step.description}</p>
                  </div>
                </div>
              </motion.div>
            );
          })}

          {/* Current Step Highlight */}
          <AnimatePresence>
            {currentStep < steps.length && (
              <motion.div
                className="absolute bottom-6 left-6 right-6"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
              >
                <div className="bg-black/60 backdrop-blur-xl rounded-2xl p-6 border border-white/10">
                  <div className="flex items-center gap-4">
                    <div className={`w-12 h-12 rounded-full bg-gradient-to-r ${steps[currentStep].color} flex items-center justify-center`}>
                      {React.createElement(steps[currentStep].icon, { className: "w-6 h-6 text-white" })}
                    </div>
                    <div>
                      <h3 className="text-xl font-bold text-white mb-2">{steps[currentStep].title}</h3>
                      <p className="text-gray-300">{steps[currentStep].description}</p>
                    </div>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Layer Architecture */}
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6"
        >
          {[
            {
              layer: 'Layer 4',
              title: 'System Interface',
              components: ['React Frontend', 'FastAPI Backend', 'PostgreSQL', 'Monitoring'],
              color: 'from-blue-500 to-cyan-500',
              icon: Shield
            },
            {
              layer: 'Layer 3',
              title: 'Agentic Intelligence',
              components: ['LangGraph Agent', 'Routing Engine', 'Confidence Evaluator', 'Explainer'],
              color: 'from-purple-500 to-pink-500',
              icon: Brain
            },
            {
              layer: 'Layer 2',
              title: 'Model Bank',
              components: ['BG-Model', 'AV-Model', 'CM-Model', 'RR-Model', 'LL-Model', 'TM-Model'],
              color: 'from-green-500 to-emerald-500',
              icon: Network
            },
            {
              layer: 'Layer 1',
              title: 'Input Processing',
              components: ['Video Decoder', 'Frame Sampler', 'Face Detector', 'Audio Extractor'],
              color: 'from-orange-500 to-red-500',
              icon: Cpu
            }
          ].map((layer, index) => {
            const Icon = layer.icon;
            return (
              <motion.div
                key={layer.layer}
                className="bg-black/20 backdrop-blur-sm rounded-2xl p-6 border border-white/10 hover:border-white/20 transition-all duration-300 hover:scale-105"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.7 + index * 0.1 }}
              >
                <div className="flex items-center gap-3 mb-4">
                  <div className={`w-10 h-10 rounded-lg bg-gradient-to-r ${layer.color} flex items-center justify-center`}>
                    <Icon className="w-5 h-5 text-white" />
                  </div>
                  <div>
                    <div className="text-sm text-gray-400">{layer.layer}</div>
                    <h3 className="text-lg font-bold text-white">{layer.title}</h3>
                  </div>
                </div>
                <div className="space-y-2">
                  {layer.components.map((component, idx) => (
                    <div key={idx} className="flex items-center gap-2">
                      <div className="w-2 h-2 rounded-full bg-gradient-to-r from-cyan-400 to-purple-400"></div>
                      <span className="text-sm text-gray-300">{component}</span>
                    </div>
                  ))}
                </div>
              </motion.div>
            );
          })}
        </motion.div>
      </div>
    </div>
  );
};

export default ArchitectureJourney;