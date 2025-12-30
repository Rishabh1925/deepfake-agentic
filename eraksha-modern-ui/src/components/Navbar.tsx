import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Camera, Moon, Sun, Network, Cpu, GitBranch } from 'lucide-react';
import { useTheme } from '../context/ThemeContext';

const Navbar = () => {
  const location = useLocation();
  const { theme, toggleTheme } = useTheme();

  const isActive = (path: string) => location.pathname === path;

  return (
    <nav className="fixed top-6 left-1/2 -translate-x-1/2 z-50 w-[95%] max-w-7xl">
      <div className="bg-black/20 backdrop-blur-xl border border-white/10 rounded-2xl shadow-2xl px-8 py-4">
        <div className="flex items-center justify-between">
          {/* Logo */}
          <Link to="/" className="flex items-center gap-3">
            <div className="relative">
              <Camera className="w-8 h-8 text-cyan-400 stroke-2" strokeWidth={1.5} />
              <div className="absolute -top-1 -right-1 w-3 h-3 bg-purple-500 rounded-full animate-pulse"></div>
            </div>
            <div>
              <span className="text-2xl font-bold bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent">
                E-Raksha
              </span>
              <div className="text-xs text-gray-400 -mt-1">Neural Architecture</div>
            </div>
          </Link>

          {/* Navigation Links */}
          <div className="flex items-center gap-6">
            <Link
              to="/"
              className={`text-sm font-medium transition-all duration-300 ${
                isActive('/')
                  ? 'text-cyan-400 scale-110'
                  : 'text-gray-300 hover:text-white hover:scale-105'
              }`}
            >
              Home
            </Link>
            
            {/* Architecture Dropdown */}
            <div className="relative group">
              <button className="flex items-center gap-2 text-sm font-medium text-gray-300 hover:text-white transition-all duration-300 hover:scale-105">
                <Network className="w-4 h-4" />
                Architecture
              </button>
              <div className="absolute top-full left-0 mt-2 w-48 bg-black/80 backdrop-blur-xl border border-white/10 rounded-xl shadow-2xl opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-300">
                <Link
                  to="/architecture"
                  className="block px-4 py-3 text-sm text-gray-300 hover:text-cyan-400 hover:bg-white/5 transition-colors"
                >
                  <div className="flex items-center gap-2">
                    <GitBranch className="w-4 h-4" />
                    System Journey
                  </div>
                </Link>
                <Link
                  to="/model-flow"
                  className="block px-4 py-3 text-sm text-gray-300 hover:text-purple-400 hover:bg-white/5 transition-colors"
                >
                  <div className="flex items-center gap-2">
                    <Cpu className="w-4 h-4" />
                    Model Flow
                  </div>
                </Link>
                <Link
                  to="/data-pipeline"
                  className="block px-4 py-3 text-sm text-gray-300 hover:text-pink-400 hover:bg-white/5 transition-colors rounded-b-xl"
                >
                  <div className="flex items-center gap-2">
                    <Network className="w-4 h-4" />
                    Data Pipeline
                  </div>
                </Link>
              </div>
            </div>

            <Link
              to="/workbench"
              className={`text-sm font-medium transition-all duration-300 ${
                isActive('/workbench')
                  ? 'text-purple-400 scale-110'
                  : 'text-gray-300 hover:text-white hover:scale-105'
              }`}
            >
              Analysis
            </Link>
            <Link
              to="/dashboard"
              className={`text-sm font-medium transition-all duration-300 ${
                isActive('/dashboard')
                  ? 'text-pink-400 scale-110'
                  : 'text-gray-300 hover:text-white hover:scale-105'
              }`}
            >
              Dashboard
            </Link>
            <Link
              to="/analytics"
              className={`text-sm font-medium transition-all duration-300 ${
                isActive('/analytics')
                  ? 'text-green-400 scale-110'
                  : 'text-gray-300 hover:text-white hover:scale-105'
              }`}
            >
              Statistics
            </Link>

            {/* Theme Toggle */}
            <button
              onClick={toggleTheme}
              className="p-2 rounded-lg hover:bg-white/10 transition-all duration-300 hover:scale-110"
              aria-label="Toggle theme"
            >
              {theme === 'light' ? (
                <Moon className="w-5 h-5 text-gray-300" />
              ) : (
                <Sun className="w-5 h-5 text-gray-300" />
              )}
            </button>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;