import React from 'react';
import { TrendingUp, Activity, BarChart3 } from 'lucide-react';
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const Analytics = () => {
  const stats = [
    {
      label: 'Total Videos',
      value: '1,247',
      icon: <BarChart3 className="w-5 h-5" />,
      change: '+156',
    },
    {
      label: 'Detection Confidence',
      value: '94.9%',
      icon: <TrendingUp className="w-5 h-5" />,
      change: '+4.2%',
    },
    {
      label: 'Avg Processing',
      value: '2.1s',
      icon: <Activity className="w-5 h-5" />,
      change: '-0.8s',
    },
  ];

  const trendData = [
    { month: 'Jan', processed: 850, accuracy: 89 },
    { month: 'Feb', processed: 920, accuracy: 91 },
    { month: 'Mar', processed: 1050, accuracy: 92.5 },
    { month: 'Apr', processed: 1180, accuracy: 93.8 },
    { month: 'May', processed: 1247, accuracy: 94.9 },
  ];

  const dailyData = [
    { day: 'Mon', scans: 45 },
    { day: 'Tue', scans: 52 },
    { day: 'Wed', scans: 38 },
    { day: 'Thu', scans: 61 },
    { day: 'Fri', scans: 55 },
    { day: 'Sat', scans: 42 },
    { day: 'Sun', scans: 35 },
  ];

  return (
    <div className="min-h-screen pt-32 pb-20">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="mb-12">
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
            Interceptor Statistics
          </h1>
          <p className="text-gray-600 dark:text-gray-400">
            Monitor detection confidence, processing times, and model performance across all 6 specialist networks
          </p>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
          {stats.map((stat, index) => (
            <div
              key={index}
              className="bg-white/50 dark:bg-gray-900/50 backdrop-blur-md border border-gray-200 dark:border-gray-800 rounded-2xl p-6"
            >
              <div className="flex items-center justify-between mb-4">
                <div className="w-10 h-10 bg-blue-100 dark:bg-blue-900/30 rounded-xl flex items-center justify-center text-blue-600 dark:text-blue-400">
                  {stat.icon}
                </div>
                <div className="text-right">
                  <p className="text-sm text-green-600 dark:text-green-400 font-semibold">
                    {stat.change}
                  </p>
                </div>
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">{stat.label}</p>
              <p className="text-3xl font-bold text-gray-900 dark:text-white">{stat.value}</p>
            </div>
          ))}
        </div>

        {/* Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-12">
          {/* Trend Chart */}
          <div className="bg-white/50 dark:bg-gray-900/50 backdrop-blur-md border border-gray-200 dark:border-gray-800 rounded-2xl p-6">
            <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-6">
              Monthly Trend
            </h2>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={trendData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" className="dark:stroke-gray-800" />
                <XAxis dataKey="month" stroke="#6B7280" />
                <YAxis stroke="#6B7280" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#FFFFFF',
                    border: '1px solid #E5E7EB',
                    borderRadius: '8px',
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="processed"
                  stroke="#3B82F6"
                  strokeWidth={2}
                  dot={{ fill: '#3B82F6' }}
                />
                <Line
                  type="monotone"
                  dataKey="accuracy"
                  stroke="#10B981"
                  strokeWidth={2}
                  dot={{ fill: '#10B981' }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Daily Scans */}
          <div className="bg-white/50 dark:bg-gray-900/50 backdrop-blur-md border border-gray-200 dark:border-gray-800 rounded-2xl p-6">
            <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-6">
              Weekly Activity
            </h2>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={dailyData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" className="dark:stroke-gray-800" />
                <XAxis dataKey="day" stroke="#6B7280" />
                <YAxis stroke="#6B7280" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#FFFFFF',
                    border: '1px solid #E5E7EB',
                    borderRadius: '8px',
                  }}
                />
                <Area
                  type="monotone"
                  dataKey="scans"
                  stroke="#3B82F6"
                  fill="#3B82F6"
                  fillOpacity={0.2}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Real-time Stats */}
        <div className="bg-white/50 dark:bg-gray-900/50 backdrop-blur-md border border-gray-200 dark:border-gray-800 rounded-2xl p-8">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
            Model Performance
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
            <div className="bg-green-50/50 dark:bg-green-900/20 backdrop-blur-md rounded-xl p-4 border border-green-200 dark:border-green-800">
              <p className="text-xs text-green-600 dark:text-green-400 mb-1">BG-Model</p>
              <p className="text-xl font-bold text-gray-900 dark:text-white">86.25%</p>
            </div>
            <div className="bg-blue-50/50 dark:bg-blue-900/20 backdrop-blur-md rounded-xl p-4 border border-blue-200 dark:border-blue-800">
              <p className="text-xs text-blue-600 dark:text-blue-400 mb-1">AV-Model</p>
              <p className="text-xl font-bold text-gray-900 dark:text-white">93.0%</p>
            </div>
            <div className="bg-purple-50/50 dark:bg-purple-900/20 backdrop-blur-md rounded-xl p-4 border border-purple-200 dark:border-purple-800">
              <p className="text-xs text-purple-600 dark:text-purple-400 mb-1">CM-Model</p>
              <p className="text-xl font-bold text-gray-900 dark:text-white">80.83%</p>
            </div>
            <div className="bg-orange-50/50 dark:bg-orange-900/20 backdrop-blur-md rounded-xl p-4 border border-orange-200 dark:border-orange-800">
              <p className="text-xs text-orange-600 dark:text-orange-400 mb-1">RR-Model</p>
              <p className="text-xl font-bold text-gray-900 dark:text-white">85.0%</p>
            </div>
            <div className="bg-cyan-50/50 dark:bg-cyan-900/20 backdrop-blur-md rounded-xl p-4 border border-cyan-200 dark:border-cyan-800">
              <p className="text-xs text-cyan-600 dark:text-cyan-400 mb-1">LL-Model</p>
              <p className="text-xl font-bold text-gray-900 dark:text-white">93.42%</p>
            </div>
            <div className="bg-pink-50/50 dark:bg-pink-900/20 backdrop-blur-md rounded-xl p-4 border border-pink-200 dark:border-pink-800">
              <p className="text-xs text-pink-600 dark:text-pink-400 mb-1">TM-Model</p>
              <p className="text-xl font-bold text-gray-900 dark:text-white">78.5%</p>
            </div>
          </div>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mt-6 pt-6 border-t border-gray-200 dark:border-gray-800">
            <div className="bg-gray-50/50 dark:bg-gray-800/50 backdrop-blur-md rounded-xl p-4">
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">Total Scans</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">1,247</p>
            </div>
            <div className="bg-gray-50/50 dark:bg-gray-800/50 backdrop-blur-md rounded-xl p-4">
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">Avg Processing</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">2.1s</p>
            </div>
            <div className="bg-gray-50/50 dark:bg-gray-800/50 backdrop-blur-md rounded-xl p-4">
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">Total Parameters</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">47.2M</p>
            </div>
            <div className="bg-gray-50/50 dark:bg-gray-800/50 backdrop-blur-md rounded-xl p-4">
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">Memory Usage</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">512MB</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Analytics;