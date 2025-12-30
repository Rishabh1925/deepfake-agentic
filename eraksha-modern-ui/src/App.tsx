import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider } from './context/ThemeContext';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import AnalysisWorkbench from './pages/AnalysisWorkbench';
import Dashboard from './pages/Dashboard';
import Analytics from './pages/Analytics';
import Contact from './pages/Contact';
import FAQ from './pages/FAQ';
import ArchitectureJourney from './pages/ArchitectureJourney';
import ModelFlow from './pages/ModelFlow';
import DataPipeline from './pages/DataPipeline';

export default function App() {
  return (
    <ThemeProvider>
      <Router>
        <div className="min-h-screen bg-gradient-to-br from-slate-950 via-purple-950 to-slate-950 transition-colors relative overflow-hidden">
          {/* Neural network inspired animated background */}
          <div className="fixed inset-0 opacity-20">
            <div className="absolute top-0 left-0 w-96 h-96 bg-blue-500 rounded-full mix-blend-multiply filter blur-3xl animate-pulse"></div>
            <div className="absolute top-1/4 right-0 w-80 h-80 bg-purple-500 rounded-full mix-blend-multiply filter blur-3xl animate-pulse animation-delay-1000"></div>
            <div className="absolute bottom-1/4 left-1/3 w-72 h-72 bg-cyan-500 rounded-full mix-blend-multiply filter blur-3xl animate-pulse animation-delay-2000"></div>
            <div className="absolute bottom-0 right-1/3 w-64 h-64 bg-pink-500 rounded-full mix-blend-multiply filter blur-3xl animate-pulse animation-delay-3000"></div>
          </div>
          
          {/* Neural network connection lines */}
          <div className="fixed inset-0 opacity-10">
            <svg className="w-full h-full">
              <defs>
                <linearGradient id="connectionGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%" stopColor="#3b82f6" />
                  <stop offset="50%" stopColor="#8b5cf6" />
                  <stop offset="100%" stopColor="#06b6d4" />
                </linearGradient>
              </defs>
              <g className="animate-pulse">
                <line x1="10%" y1="20%" x2="90%" y2="80%" stroke="url(#connectionGradient)" strokeWidth="1" />
                <line x1="20%" y1="10%" x2="80%" y2="90%" stroke="url(#connectionGradient)" strokeWidth="1" />
                <line x1="30%" y1="30%" x2="70%" y2="70%" stroke="url(#connectionGradient)" strokeWidth="1" />
                <line x1="40%" y1="15%" x2="60%" y2="85%" stroke="url(#connectionGradient)" strokeWidth="1" />
              </g>
            </svg>
          </div>
          
          <div className="relative z-10">
            <Navbar />
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/workbench" element={<AnalysisWorkbench />} />
              <Route path="/dashboard" element={<Dashboard />} />
              <Route path="/analytics" element={<Analytics />} />
              <Route path="/architecture" element={<ArchitectureJourney />} />
              <Route path="/model-flow" element={<ModelFlow />} />
              <Route path="/data-pipeline" element={<DataPipeline />} />
              <Route path="/contact" element={<Contact />} />
              <Route path="/faq" element={<FAQ />} />
            </Routes>
          </div>
        </div>
      </Router>
    </ThemeProvider>
  );
}