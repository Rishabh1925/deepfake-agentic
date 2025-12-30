import React from 'react';
import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import { 
  Shield, 
  Zap, 
  Eye, 
  Brain, 
  Network, 
  Cpu, 
  ArrowRight,
  Play,
  GitBranch,
  Database
} from 'lucide-react';

const Home = () => {
  return (
    <div className="min-h-screen pt-32 pb-20">
      {/* Hero Section */}
      <section className="px-6 mb-20">
        <div className="max-w-7xl mx-auto text-center">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            <h1 className="text-7xl md:text-8xl font-bold mb-8">
              <span className="bg-gradient-to-r from-cyan-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
                E-Raksha
              </span>
            </h1>
            <p className="text-2xl md:text-3xl text-gray-300 mb-6 max-w-4xl mx-auto">
              Agentic AI for Deepfake Detection & Authenticity Verification
            </p>
            <p className="text-lg text-gray-400 mb-12 max-w-3xl mx-auto">
              Experience the complete neural architecture journey from video upload to deepfake detection 
              through our four-layer system with intelligent agentic routing and specialist models.
            </p>
            
            {/* CTA Buttons */}
            <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
              <Link
                to="/architecture"
                className="group flex items-center gap-3 px-8 py-4 bg-gradient-to-r from-cyan-500 to-purple-500 rounded-2xl text-white font-semibold text-lg hover:scale-105 transition-all duration-300 shadow-2xl"
              >
                <GitBranch className="w-6 h-6" />
                Explore Architecture
                <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
              </Link>
              <Link
                to="/workbench"
                className="group flex items-center gap-3 px-8 py-4 bg-black/30 backdrop-blur-sm border border-white/20 rounded-2xl text-white font-semibold text-lg hover:bg-white/10 transition-all duration-300"
              >
                <Play className="w-6 h-6" />
                Try Detection
                <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
              </Link>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Architecture Overview */}
      <section className="px-6 mb-20">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl md:text-5xl font-bold text-white mb-6">
              Four-Layer Neural Architecture
            </h2>
            <p className="text-xl text-gray-300 max-w-3xl mx-auto">
              Discover how data flows through our sophisticated multi-layer system, 
              from input processing to intelligent agentic decision making.
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {[
              {
                layer: 'Layer 4',
                title: 'System Interface',
                description: 'React frontend, FastAPI backend, and monitoring systems',
                icon: Shield,
                color: 'from-blue-500 to-cyan-500',
                link: '/architecture'
              },
              {
                layer: 'Layer 3',
                title: 'Agentic Intelligence',
                description: 'LangGraph agent with intelligent routing and confidence evaluation',
                icon: Brain,
                color: 'from-purple-500 to-pink-500',
                link: '/architecture'
              },
              {
                layer: 'Layer 2',
                title: 'Model Bank',
                description: 'Six specialist models: BG, AV, CM, RR, LL, and TM models',
                icon: Network,
                color: 'from-green-500 to-emerald-500',
                link: '/model-flow'
              },
              {
                layer: 'Layer 1',
                title: 'Input Processing',
                description: 'Video decoder, frame sampler, face detector, and audio extractor',
                icon: Cpu,
                color: 'from-orange-500 to-red-500',
                link: '/data-pipeline'
              }
            ].map((layer, index) => {
              const Icon = layer.icon;
              return (
                <motion.div
                  key={layer.layer}
                  initial={{ opacity: 0, y: 40 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.6, delay: 0.4 + index * 0.1 }}
                >
                  <Link
                    to={layer.link}
                    className="group block bg-black/20 backdrop-blur-sm rounded-3xl p-8 border border-white/10 hover:border-white/30 transition-all duration-300 hover:scale-105 hover:bg-white/5"
                  >
                    <div className="flex items-center gap-4 mb-6">
                      <div className={`w-16 h-16 rounded-2xl bg-gradient-to-r ${layer.color} flex items-center justify-center group-hover:scale-110 transition-transform duration-300`}>
                        <Icon className="w-8 h-8 text-white" />
                      </div>
                      <div>
                        <div className="text-sm text-gray-400 mb-1">{layer.layer}</div>
                        <h3 className="text-xl font-bold text-white group-hover:text-transparent group-hover:bg-gradient-to-r group-hover:from-cyan-400 group-hover:to-purple-400 group-hover:bg-clip-text transition-all duration-300">
                          {layer.title}
                        </h3>
                      </div>
                    </div>
                    <p className="text-gray-300 group-hover:text-gray-200 transition-colors duration-300">
                      {layer.description}
                    </p>
                    <div className="flex items-center gap-2 mt-4 text-cyan-400 group-hover:text-purple-400 transition-colors duration-300">
                      <span className="text-sm font-medium">Explore</span>
                      <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform duration-300" />
                    </div>
                  </Link>
                </motion.div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Model Showcase */}
      <section className="px-6 mb-20">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.6 }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl md:text-5xl font-bold text-white mb-6">
              Specialist Model Bank
            </h2>
            <p className="text-xl text-gray-300 max-w-3xl mx-auto">
              Six specialized neural networks working together to detect different types of deepfake artifacts
            </p>
          </motion.div>

          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
            {[
              { name: 'BG-Model', desc: 'Baseline Generalist', params: '2.1M', accuracy: '86.25%', color: 'from-blue-500 to-cyan-500' },
              { name: 'AV-Model', desc: 'Audio-Visual', params: '15.8M', accuracy: '93.0%', color: 'from-purple-500 to-pink-500' },
              { name: 'CM-Model', desc: 'Compression', params: '11.7M', accuracy: '80.83%', color: 'from-orange-500 to-red-500' },
              { name: 'RR-Model', desc: 'Re-recording', params: '11.7M', accuracy: '85.0%', color: 'from-pink-500 to-rose-500' },
              { name: 'LL-Model', desc: 'Low-light', params: '11.7M', accuracy: '93.42%', color: 'from-green-500 to-emerald-500' },
              { name: 'TM-Model', desc: 'Temporal', params: '14.2M', accuracy: '78.5%', color: 'from-indigo-500 to-purple-500' }
            ].map((model, index) => (
              <motion.div
                key={model.name}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.5, delay: 0.8 + index * 0.1 }}
                className="group bg-black/20 backdrop-blur-sm rounded-2xl p-6 border border-white/10 hover:border-white/30 transition-all duration-300 hover:scale-105"
              >
                <div className={`w-12 h-12 rounded-xl bg-gradient-to-r ${model.color} flex items-center justify-center mb-4 group-hover:scale-110 transition-transform duration-300`}>
                  <Cpu className="w-6 h-6 text-white" />
                </div>
                <h3 className="text-lg font-bold text-white mb-2">{model.name}</h3>
                <p className="text-sm text-gray-400 mb-3">{model.desc}</p>
                <div className="space-y-1">
                  <div className="flex justify-between text-xs">
                    <span className="text-gray-400">Params:</span>
                    <span className="text-cyan-400 font-mono">{model.params}</span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-gray-400">Accuracy:</span>
                    <span className="text-green-400 font-mono">{model.accuracy}</span>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 1.4 }}
            className="text-center mt-12"
          >
            <Link
              to="/model-flow"
              className="inline-flex items-center gap-3 px-8 py-4 bg-gradient-to-r from-purple-500 to-pink-500 rounded-2xl text-white font-semibold hover:scale-105 transition-all duration-300"
            >
              <Network className="w-6 h-6" />
              Explore Model Architectures
              <ArrowRight className="w-5 h-5" />
            </Link>
          </motion.div>
        </div>
      </section>

      {/* Data Flow Preview */}
      <section className="px-6">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 1.0 }}
            className="bg-black/20 backdrop-blur-sm rounded-3xl border border-white/10 p-12 text-center"
          >
            <h2 className="text-4xl md:text-5xl font-bold text-white mb-6">
              Watch Data Flow in Real-Time
            </h2>
            <p className="text-xl text-gray-300 mb-8 max-w-3xl mx-auto">
              Experience the complete journey from video upload to deepfake detection result 
              with animated visualizations of data flowing through our neural architecture.
            </p>
            
            {/* Mini Pipeline Preview */}
            <div className="flex items-center justify-center gap-4 mb-8 overflow-x-auto">
              {['Upload', 'Process', 'Detect', 'Route', 'Analyze', 'Result'].map((step, index) => (
                <div key={step} className="flex items-center">
                  <motion.div
                    className="w-16 h-16 rounded-2xl bg-gradient-to-r from-cyan-500 to-purple-500 flex items-center justify-center"
                    animate={{ scale: [1, 1.1, 1] }}
                    transition={{ duration: 2, delay: index * 0.3, repeat: Infinity }}
                  >
                    <span className="text-white font-bold text-sm">{step}</span>
                  </motion.div>
                  {index < 5 && (
                    <motion.div
                      className="w-8 h-0.5 bg-gradient-to-r from-cyan-400 to-purple-400 mx-2"
                      initial={{ scaleX: 0 }}
                      animate={{ scaleX: 1 }}
                      transition={{ duration: 0.5, delay: index * 0.3 + 1 }}
                    />
                  )}
                </div>
              ))}
            </div>
            
            <Link
              to="/data-pipeline"
              className="inline-flex items-center gap-3 px-8 py-4 bg-gradient-to-r from-cyan-500 to-purple-500 rounded-2xl text-white font-semibold hover:scale-105 transition-all duration-300"
            >
              <Database className="w-6 h-6" />
              Explore Data Pipeline
              <ArrowRight className="w-5 h-5" />
            </Link>
          </motion.div>
        </div>
      </section>
    </div>
  );
};

export default Home;