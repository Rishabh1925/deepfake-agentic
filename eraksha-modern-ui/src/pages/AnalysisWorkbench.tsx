import React, { useState, useCallback, useEffect, useRef } from 'react';
import { Upload, FileVideo, CheckCircle2, XCircle, Download, Activity } from 'lucide-react';
import { Progress } from '../components/ui/progress';
import { useArchitecture } from '../context/ArchitectureContext';
import ModelProgressCanvas from '../components/ModelProgressCanvas';

const AnalysisWorkbench = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [analysisResult, setAnalysisResult] = useState<any>(null);
  const [currentStage, setCurrentStage] = useState('');
  const visualizationRef = useRef<HTMLDivElement>(null);

  const {
    setCurrentPage,
    setProcessingStage,
    activateModel,
    resetFlow,
  } = useArchitecture();

  useEffect(() => {
    setCurrentPage('workbench');
    return () => resetFlow();
  }, [setCurrentPage, resetFlow]);

  // Auto-scroll to visualization when analysis starts
  useEffect(() => {
    if (isAnalyzing && visualizationRef.current) {
      visualizationRef.current.scrollIntoView({ 
        behavior: 'smooth', 
        block: 'start' 
      });
    }
  }, [isAnalyzing]);

  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const files = e.dataTransfer.files;
    if (files && files[0]) {
      setSelectedFile(files[0]);
    }
  }, []);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files[0]) {
      setSelectedFile(files[0]);
    }
  }, []);

  const simulateAnalysisWithVisualization = async () => {
    setIsAnalyzing(true);
    setProgress(0);
    resetFlow();

    // Stage 1: Video Ingestion
    setCurrentStage('Video Ingestion');
    setProcessingStage('Video Ingestion');
    activateModel('video-input');
    activateModel('video-decoder');
    await new Promise(resolve => setTimeout(resolve, 800));
    setProgress(10);

    // Stage 2: Frame Extraction
    setCurrentStage('Frame Extraction');
    setProcessingStage('Frame Extraction');
    activateModel('frame-sampler');
    activateModel('face-detector');
    activateModel('audio-extractor');
    await new Promise(resolve => setTimeout(resolve, 1000));
    setProgress(25);

    // Stage 3: Baseline Inference
    setCurrentStage('Baseline Model Inference');
    setProcessingStage('Baseline Model Inference');
    activateModel('bg-model');
    await new Promise(resolve => setTimeout(resolve, 1200));
    setProgress(40);

    // Stage 4: Routing Decision
    setCurrentStage('Intelligent Routing');
    setProcessingStage('Intelligent Routing');
    activateModel('routing-engine');
    activateModel('langgraph');
    await new Promise(resolve => setTimeout(resolve, 800));
    setProgress(50);

    // Stage 5: Specialist Models (Conditional)
    setCurrentStage('Specialist Model Analysis');
    setProcessingStage('Specialist Model Analysis');
    const specialists = ['av-model', 'cm-model', 'll-model', 'tm-model'];
    for (const specialist of specialists) {
      activateModel(specialist);
      await new Promise(resolve => setTimeout(resolve, 600));
      setProgress(prev => Math.min(prev + 5, 75));
    }

    // Stage 6: Aggregation
    setCurrentStage('Prediction Aggregation');
    setProcessingStage('Prediction Aggregation');
    activateModel('aggregator');
    await new Promise(resolve => setTimeout(resolve, 800));
    setProgress(85);

    // Stage 7: Bias Correction
    setCurrentStage('Bias Correction');
    setProcessingStage('Bias Correction');
    activateModel('bias-correction');
    await new Promise(resolve => setTimeout(resolve, 600));
    setProgress(90);

    // Stage 8: Explanation Generation
    setCurrentStage('Explanation Generation');
    setProcessingStage('Explanation Generation');
    activateModel('explainer');
    await new Promise(resolve => setTimeout(resolve, 800));
    setProgress(95);

    // Stage 9: API Response
    setCurrentStage('Generating Response');
    setProcessingStage('Generating Response');
    activateModel('fastapi');
    activateModel('react-ui');
    activateModel('api-response');
    await new Promise(resolve => setTimeout(resolve, 600));
    setProgress(100);

    // Set results
    setAnalysisResult({
      prediction: 'fake',
      confidence: 0.873,
      best_model: 'AV-Model',
      specialists_used: ['AV-Model', 'CM-Model', 'LL-Model', 'TM-Model'],
      processing_time: 2.8,
      explanation: 'This video is classified as MANIPULATED (FAKE) with 87.3% confidence. Primary analysis performed by the audio-visual specialist model. Audio-visual synchronization inconsistencies detected, suggesting potential deepfake manipulation.',
      filename: selectedFile?.name,
      file_size: selectedFile?.size,
      bias_correction: true,
    });

    setIsAnalyzing(false);
    setCurrentStage('Analysis Complete');
    setProcessingStage('Analysis Complete');
  };

  const analyzeVideo = async () => {
    if (!selectedFile) return;
    await simulateAnalysisWithVisualization();
  };

  const handleReset = () => {
    setSelectedFile(null);
    setAnalysisResult(null);
    setProgress(0);
    setCurrentStage('');
    resetFlow();
  };

  return (
    <div className="min-h-screen pt-32 pb-20">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="mb-12 text-center">
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
            Video Analysis
          </h1>
          <p className="text-gray-600 dark:text-gray-400 max-w-2xl mx-auto">
            Upload your video for instant deepfake detection using our advanced agentic AI system. 
            Supports MP4, AVI, MOV, and WebM up to 100MB.
          </p>
        </div>

        {/* Upload Section */}
        <div className="mb-12">
          <div
            onDragEnter={handleDragEnter}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className={`relative border-2 border-dashed rounded-2xl p-16 transition-all backdrop-blur-md ${
              isDragging
                ? 'border-blue-500 bg-blue-50/50 dark:bg-blue-900/20'
                : 'border-gray-300 dark:border-gray-700 bg-white/50 dark:bg-gray-900/50'
            }`}
          >
            <input
              type="file"
              accept="video/mp4,video/avi,video/mov,video/webm"
              onChange={handleFileSelect}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
            />
            <div className="text-center">
              {selectedFile ? (
                <div className="flex flex-col items-center">
                  <FileVideo className="w-16 h-16 text-blue-600 dark:text-blue-400 mb-4" />
                  <p className="text-lg text-gray-900 dark:text-white mb-2">{selectedFile.name}</p>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
                  </p>
                </div>
              ) : (
                <div className="flex flex-col items-center">
                  <Upload className="w-16 h-16 text-gray-400 dark:text-gray-500 mb-4" />
                  <p className="text-lg text-gray-900 dark:text-white mb-2">
                    Drag and drop your video or click to browse
                  </p>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    Maximum file size: 100MB
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* Action Buttons */}
          {!isAnalyzing && !analysisResult && (
            <div className="flex gap-4 mt-6 justify-center">
              <button
                onClick={analyzeVideo}
                disabled={!selectedFile}
                className={`px-8 py-3 rounded-xl transition-colors shadow-lg ${
                  selectedFile
                    ? 'bg-blue-600 hover:bg-blue-700 text-white cursor-pointer'
                    : 'bg-gray-300 dark:bg-gray-700 text-gray-500 dark:text-gray-400 cursor-not-allowed'
                }`}
              >
                Analyze Video
              </button>
              {selectedFile && (
                <button
                  onClick={handleReset}
                  className="px-8 py-3 bg-white/50 dark:bg-gray-900/50 backdrop-blur-md hover:bg-white/70 dark:hover:bg-gray-900/70 text-gray-900 dark:text-white border border-gray-200 dark:border-gray-800 rounded-xl transition-colors"
                >
                  Clear
                </button>
              )}
            </div>
          )}
        </div>

        {/* Processing Progress */}
        {isAnalyzing && (
          <div className="mb-8">
            <div className="bg-white/50 dark:bg-gray-900/50 backdrop-blur-md border border-gray-200 dark:border-gray-800 rounded-2xl p-8">
              <div className="flex items-center gap-3 mb-6">
                <Activity className="w-6 h-6 text-blue-600 dark:text-blue-400 animate-pulse" />
                <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
                  Analyzing Video
                </h2>
              </div>
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between text-sm text-gray-600 dark:text-gray-400 mb-2">
                    <span className="flex items-center gap-2">
                      <span className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></span>
                      {currentStage || 'Processing with agentic system...'}
                    </span>
                    <span>{progress}%</span>
                  </div>
                  <Progress value={progress} className="h-3" />
                </div>
                <div className="text-sm text-gray-500 dark:text-gray-400 text-center">
                  Watch the real-time model activation and data flow below ↓
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Full-Width Model Progress Visualization */}
        {(isAnalyzing || analysisResult) && (
          <div ref={visualizationRef} className="mb-12">
            <div className="bg-white/50 dark:bg-gray-900/50 backdrop-blur-md border border-gray-200 dark:border-gray-800 rounded-2xl p-8">
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                  <Activity className="w-6 h-6 text-blue-600 dark:text-blue-400" />
                  <h3 className="text-xl font-semibold text-gray-900 dark:text-white">
                    E-Raksha Processing Pipeline
                  </h3>
                </div>
                <div className="text-sm text-gray-500 dark:text-gray-400">
                  {isAnalyzing ? 'Live Processing' : 'Analysis Complete'}
                </div>
              </div>
              
              <div className="relative h-[500px] rounded-xl overflow-hidden bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800">
                <ModelProgressCanvas />
              </div>
              
              {/* Enhanced Legend */}
              <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="flex items-center gap-3 p-3 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
                  <div className="w-4 h-4 bg-purple-500 rounded-full shadow-lg"></div>
                  <div>
                    <div className="font-medium text-purple-700 dark:text-purple-300 text-sm">Input Processing</div>
                    <div className="text-xs text-purple-600 dark:text-purple-400">Video & Audio</div>
                  </div>
                </div>
                <div className="flex items-center gap-3 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                  <div className="w-4 h-4 bg-blue-500 rounded-full shadow-lg"></div>
                  <div>
                    <div className="font-medium text-blue-700 dark:text-blue-300 text-sm">Model Bank</div>
                    <div className="text-xs text-blue-600 dark:text-blue-400">AI Models</div>
                  </div>
                </div>
                <div className="flex items-center gap-3 p-3 bg-violet-50 dark:bg-violet-900/20 rounded-lg">
                  <div className="w-4 h-4 bg-violet-500 rounded-full shadow-lg"></div>
                  <div>
                    <div className="font-medium text-violet-700 dark:text-violet-300 text-sm">Agentic Intelligence</div>
                    <div className="text-xs text-violet-600 dark:text-violet-400">Smart Routing</div>
                  </div>
                </div>
                <div className="flex items-center gap-3 p-3 bg-cyan-50 dark:bg-cyan-900/20 rounded-lg">
                  <div className="w-4 h-4 bg-cyan-500 rounded-full shadow-lg"></div>
                  <div>
                    <div className="font-medium text-cyan-700 dark:text-cyan-300 text-sm">Output Generation</div>
                    <div className="text-xs text-cyan-600 dark:text-cyan-400">Results & API</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Results Section */}
        {analysisResult && (
          <div className="space-y-8">
            {/* Results Header */}
            <div className="bg-white/50 dark:bg-gray-900/50 backdrop-blur-md border border-gray-200 dark:border-gray-800 rounded-2xl p-8">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
                  Analysis Results
                </h2>
                <button
                  onClick={handleReset}
                  className="px-4 py-2 bg-gray-100/50 dark:bg-gray-800/50 backdrop-blur-md hover:bg-gray-200/50 dark:hover:bg-gray-700/50 text-gray-900 dark:text-white rounded-lg transition-colors"
                >
                  New Analysis
                </button>
              </div>

              <div className="flex items-center gap-4 mb-6">
                <div
                  className={`w-16 h-16 rounded-full flex items-center justify-center ${
                    analysisResult.prediction === 'fake'
                      ? 'bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400'
                      : 'bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400'
                  }`}
                >
                  {analysisResult.prediction === 'fake' ? (
                    <XCircle className="w-8 h-8" />
                  ) : (
                    <CheckCircle2 className="w-8 h-8" />
                  )}
                </div>
                <div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">Prediction</p>
                  <p className="text-2xl font-bold text-gray-900 dark:text-white capitalize">
                    {analysisResult.prediction}
                  </p>
                </div>
              </div>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                <div className="bg-gray-50/50 dark:bg-gray-800/50 backdrop-blur-md rounded-xl p-4">
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">Confidence</p>
                  <p className="text-xl font-semibold text-gray-900 dark:text-white">
                    {(analysisResult.confidence * 100).toFixed(1)}%
                  </p>
                </div>
                <div className="bg-gray-50/50 dark:bg-gray-800/50 backdrop-blur-md rounded-xl p-4">
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">Best Model</p>
                  <p className="text-xl font-semibold text-gray-900 dark:text-white">
                    {analysisResult.best_model}
                  </p>
                </div>
                <div className="bg-gray-50/50 dark:bg-gray-800/50 backdrop-blur-md rounded-xl p-4">
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">Specialists</p>
                  <p className="text-xl font-semibold text-gray-900 dark:text-white">
                    {analysisResult.specialists_used?.length || 1}
                  </p>
                </div>
                <div className="bg-gray-50/50 dark:bg-gray-800/50 backdrop-blur-md rounded-xl p-4">
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">Time</p>
                  <p className="text-xl font-semibold text-gray-900 dark:text-white">
                    {analysisResult.processing_time}s
                  </p>
                </div>
              </div>

              <div className="bg-blue-50/50 dark:bg-blue-900/20 backdrop-blur-md rounded-xl p-4">
                <p className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
                  Explanation
                </p>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  {analysisResult.explanation}
                </p>
                {analysisResult.bias_correction && (
                  <div className="mt-3 px-3 py-1 bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400 rounded-full text-sm inline-block">
                    ✓ Bias correction applied
                  </div>
                )}
              </div>
            </div>

            <div className="flex gap-4 justify-center">
              <button className="flex items-center gap-2 px-6 py-3 bg-white/50 dark:bg-gray-900/50 backdrop-blur-md hover:bg-white/70 dark:hover:bg-gray-900/70 text-gray-900 dark:text-white border border-gray-200 dark:border-gray-800 rounded-xl transition-colors">
                <Download className="w-4 h-4" />
                Download Report
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default AnalysisWorkbench;