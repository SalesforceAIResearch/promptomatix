import React, { useState } from 'react';
import PromptOptimizer from './components/PromptOptimizer';
import PromptVersions from './components/PromptVersions';
import { Settings, List } from 'lucide-react';
import { Button } from './components/ui/button';
import { Popover, PopoverTrigger, PopoverContent } from './components/ui/popover';

function App() {
  const [activeTab, setActiveTab] = useState('optimizer');
  const [sessionData, setSessionData] = useState(null);
  const [versions, setVersions] = useState([]);
  const [isConfigOpen, setIsConfigOpen] = useState(false);
  const [isSessionOpen, setIsSessionOpen] = useState(false);
  const [config, setConfig] = useState({
    backend: 'dspy',
    model_name: '',
    model_api_key: '',
    model_api_base: '',
    model_provider: '',
    temperature: 0.7,
    max_tokens: 4000,
    training_data_path: '',
    validation_data_path: '',
    synthetic_data_size: 30,
    train_ratio: 0.2,
    miprov2_init_auto: null,
    miprov2_init_num_candidates: 5,
    miprov2_compile_max_bootstrapped_demos: 0,
    miprov2_compile_max_labeled_demos: 0,
    miprov2_compile_num_trials: 15,
    miprov2_compile_minibatch_size: 1,
    active_tab: 'model'
  });

  const handleVersionUpdate = (newVersion) => {
    setVersions(prevVersions => [...prevVersions, newVersion]);
  };

  const handleConfigChange = (key, value) => {
    setConfig(prev => ({
      ...prev,
      [key]: value
    }));
  };

  const handleDownloadSession = () => {
    const data = {
      timestamp: new Date().toISOString(),
      versions,
      sessionData
    };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `promptomatix-session-${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleUploadSession = async (file) => {
    try {
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const data = JSON.parse(e.target.result);
          console.log('Loaded session data:', data);
          
          // Extract data from either the root level or sessionData property
          const sessionInput = data.input || (data.sessionData && data.sessionData.input) || '';
          const sessionOptimizedPrompt = data.optimizedPrompt || (data.sessionData && data.sessionData.optimizedPrompt) || '';
          const sessionId = data.sessionId || (data.sessionData && data.sessionData.sessionId) || generateSessionId();
          const sessionComments = data.comments || (data.sessionData && data.sessionData.comments) || [];
          const sessionResponse = data.response || (data.sessionData && data.sessionData.response) || null;
          const sessionVersions = data.versions || (data.sessionData && data.sessionData.versions) || [];
          const sessionConfig = data.config || (data.sessionData && data.sessionData.config) || null;
          
          // Set session data with all fields
          const newSessionData = {
            input: sessionInput,
            optimizedPrompt: sessionOptimizedPrompt,
            sessionId: sessionId,
            comments: sessionComments,
            response: sessionResponse,
            versions: sessionVersions,
            config: sessionConfig
          };
          
          console.log('Setting new session data:', newSessionData);
          setSessionData(newSessionData);
          
          // Update versions state
          setVersions(sessionVersions);
          
          // Close session popover
          setIsSessionOpen(false);
        } catch (error) {
          console.error('Error parsing session file:', error);
          alert('Error loading session file. Please check the file format.');
        }
      };
      reader.readAsText(file);
    } catch (error) {
      console.error('Error uploading session:', error);
      alert('Error uploading session file.');
    }
  };

  // Helper function to generate a session ID if none exists
  const generateSessionId = () => {
    return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
  };

  const startNewSession = () => {
    setSessionData(null);
    setVersions([]);
    setIsSessionOpen(false);
  };

  return (
    <div className="min-h-screen bg-deep-black font-mono text-zinc-100">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="flex flex-col space-y-6">
          {/* Top Bar */}
          <div className="flex justify-between items-center">
            <div className="flex items-center space-x-4">
              <img 
                src="/images/logo1.png"
                alt="Promptomatix Logo" 
                className="h-8 w-auto"
              />
              <h1 className="text-2xl font-bold text-zinc-100 tracking-tight font-mono">Promptomatix</h1>
            </div>

            {/* Config and Session Controls */}
            <div className="flex items-center space-x-4">
              <Popover open={isConfigOpen} onOpenChange={setIsConfigOpen}>
                <PopoverTrigger asChild>
                  <button className="btn btn-blue flex items-center gap-2">
                    <Settings className="h-5 w-5" />
                    Config
                  </button>
                </PopoverTrigger>
                <PopoverContent className="w-[600px] bg-deep border border-zinc-800 shadow-lg p-6" align="end">
                  <div className="space-y-6">
                    {/* Backend Selection */}
                    <div>
                      <h3 className="text-lg font-medium text-zinc-200 mb-4">Backend Selection</h3>
                      <div className="flex gap-4">
                        <button
                          onClick={() => handleConfigChange('backend', 'dspy')}
                          className={`flex-1 p-3 rounded-lg border ${
                            config.backend === 'dspy'
                              ? 'border-indigo-500 bg-indigo-500/10 text-indigo-300'
                              : 'border-zinc-700 text-zinc-400 hover:border-zinc-600'
                          }`}
                        >
                          DSPy
                        </button>
                        <button
                          onClick={() => handleConfigChange('backend', 'adalflow')}
                          className={`flex-1 p-3 rounded-lg border ${
                            config.backend === 'adalflow'
                              ? 'border-indigo-500 bg-indigo-500/10 text-indigo-300'
                              : 'border-zinc-700 text-zinc-400 hover:border-zinc-600'
                          }`}
                          disabled
                        >
                          Adalflow (Coming Soon)
                        </button>
                      </div>
                    </div>

                    {/* Configuration Tabs */}
                    <div>
                      <div className="flex border-b border-zinc-800 mb-6">
                        <button
                          onClick={() => handleConfigChange('active_tab', 'model')}
                          className={`px-4 py-2 -mb-px text-sm font-medium ${
                            config.active_tab === 'model'
                              ? 'text-indigo-400 border-b-2 border-indigo-400'
                              : 'text-zinc-400 hover:text-zinc-300'
                          }`}
                        >
                          Model Config
                        </button>
                        <button
                          onClick={() => handleConfigChange('active_tab', 'data')}
                          className={`px-4 py-2 -mb-px text-sm font-medium ${
                            config.active_tab === 'data'
                              ? 'text-indigo-400 border-b-2 border-indigo-400'
                              : 'text-zinc-400 hover:text-zinc-300'
                          }`}
                        >
                          Data Config
                        </button>
                        <button
                          onClick={() => handleConfigChange('active_tab', 'dspy')}
                          className={`px-4 py-2 -mb-px text-sm font-medium ${
                            config.active_tab === 'dspy'
                              ? 'text-indigo-400 border-b-2 border-indigo-400'
                              : 'text-zinc-400 hover:text-zinc-300'
                          }`}
                        >
                          DSPy Config
                        </button>
                      </div>

                      {/* Model Configuration Tab */}
                      {config.active_tab === 'model' && (
                        <div className="space-y-4">
                          <div>
                            <label className="block text-sm font-medium text-zinc-400 mb-1">
                              Model Provider
                            </label>
                            <input
                              type="text"
                              value={config.model_provider}
                              onChange={(e) => handleConfigChange('model_provider', e.target.value)}
                              className="bg-deep-black border border-zinc-800 text-zinc-100 font-mono rounded-lg p-3"
                              placeholder="e.g., OpenAI, Anthropic"
                            />
                          </div>
                          <div>
                            <label className="block text-sm font-medium text-zinc-400 mb-1">
                              Model Name
                            </label>
                            <input
                              type="text"
                              value={config.model_name}
                              onChange={(e) => handleConfigChange('model_name', e.target.value)}
                              className="bg-deep-black border border-zinc-800 text-zinc-100 font-mono rounded-lg p-3"
                              placeholder="e.g., gpt-4, claude-3"
                            />
                          </div>
                          <div>
                            <label className="block text-sm font-medium text-zinc-400 mb-1">
                              API Key
                            </label>
                            <input
                              type="password"
                              value={config.model_api_key}
                              onChange={(e) => handleConfigChange('model_api_key', e.target.value)}
                              className="bg-deep-black border border-zinc-800 text-zinc-100 font-mono rounded-lg p-3"
                              placeholder="Enter your API key"
                            />
                          </div>
                          <div>
                            <label className="block text-sm font-medium text-zinc-400 mb-1">
                              Temperature
                            </label>
                            <input
                              type="number"
                              min="0"
                              max="1"
                              step="0.1"
                              value={config.temperature}
                              onChange={(e) => handleConfigChange('temperature', parseFloat(e.target.value))}
                              className="bg-deep-black border border-zinc-800 text-zinc-100 font-mono rounded-lg p-3"
                            />
                          </div>
                          <div>
                            <label className="block text-sm font-medium text-zinc-400 mb-1">
                              Max Tokens
                            </label>
                            <input
                              type="number"
                              value={config.max_tokens}
                              onChange={(e) => handleConfigChange('max_tokens', parseInt(e.target.value))}
                              className="bg-deep-black border border-zinc-800 text-zinc-100 font-mono rounded-lg p-3"
                            />
                          </div>
                        </div>
                      )}

                      {/* Data Configuration Tab */}
                      {config.active_tab === 'data' && (
                        <div className="space-y-4">
                          <div>
                            <label className="block text-sm font-medium text-zinc-400 mb-1">
                              Training Data Path
                            </label>
                            <input
                              type="text"
                              value={config.training_data_path}
                              onChange={(e) => handleConfigChange('training_data_path', e.target.value)}
                              className="bg-deep-black border border-zinc-800 text-zinc-100 font-mono rounded-lg p-3"
                              placeholder="Path to your training data"
                            />
                          </div>
                          <div>
                            <label className="block text-sm font-medium text-zinc-400 mb-1">
                              Validation Data Path
                            </label>
                            <input
                              type="text"
                              value={config.validation_data_path}
                              onChange={(e) => handleConfigChange('validation_data_path', e.target.value)}
                              className="bg-deep-black border border-zinc-800 text-zinc-100 font-mono rounded-lg p-3"
                              placeholder="Path to your validation data"
                            />
                          </div>
                        </div>
                      )}

                      {/* DSPy Configuration Tab */}
                      {config.active_tab === 'dspy' && (
                        <div className="space-y-4">
                          <div>
                            <label className="block text-sm font-medium text-zinc-400 mb-1">
                              Synthetic Data Size
                            </label>
                            <input
                              type="number"
                              value={config.synthetic_data_size}
                              onChange={(e) => handleConfigChange('synthetic_data_size', parseInt(e.target.value))}
                              className="bg-deep-black border border-zinc-800 text-zinc-100 font-mono rounded-lg p-3"
                              min="1"
                            />
                          </div>
                          <div>
                            <label className="block text-sm font-medium text-zinc-400 mb-1">
                              Train Ratio
                            </label>
                            <input
                              type="number"
                              value={config.train_ratio}
                              onChange={(e) => handleConfigChange('train_ratio', parseFloat(e.target.value))}
                              className="bg-deep-black border border-zinc-800 text-zinc-100 font-mono rounded-lg p-3"
                              min="0"
                              max="1"
                              step="0.1"
                            />
                          </div>
                          <div>
                            <label className="block text-sm font-medium text-zinc-400 mb-1">
                              MiProv2 Init Auto
                            </label>
                            <input
                              type="text"
                              value={config.miprov2_init_auto === null ? 'null' : config.miprov2_init_auto.toString()}
                              onChange={(e) => {
                                let value = e.target.value.toLowerCase();
                                if (value === 'null') value = null;
                                else if (value === 'true') value = true;
                                else if (value === 'false') value = false;
                                handleConfigChange('miprov2_init_auto', value);
                              }}
                              className="bg-deep-black border border-zinc-800 text-zinc-100 font-mono rounded-lg p-3"
                              placeholder="null, true, or false"
                            />
                          </div>
                          <div>
                            <label className="block text-sm font-medium text-zinc-400 mb-1">
                              MiProv2 Init Num Candidates
                            </label>
                            <input
                              type="number"
                              value={config.miprov2_init_num_candidates}
                              onChange={(e) => handleConfigChange('miprov2_init_num_candidates', parseInt(e.target.value))}
                              className="bg-deep-black border border-zinc-800 text-zinc-100 font-mono rounded-lg p-3"
                              min="1"
                            />
                          </div>
                          <div>
                            <label className="block text-sm font-medium text-zinc-400 mb-1">
                              MiProv2 Compile Max Bootstrapped Demos
                            </label>
                            <input
                              type="number"
                              value={config.miprov2_compile_max_bootstrapped_demos}
                              onChange={(e) => handleConfigChange('miprov2_compile_max_bootstrapped_demos', parseInt(e.target.value))}
                              className="bg-deep-black border border-zinc-800 text-zinc-100 font-mono rounded-lg p-3"
                              min="0"
                            />
                          </div>
                          <div>
                            <label className="block text-sm font-medium text-zinc-400 mb-1">
                              MiProv2 Compile Max Labeled Demos
                            </label>
                            <input
                              type="number"
                              value={config.miprov2_compile_max_labeled_demos}
                              onChange={(e) => handleConfigChange('miprov2_compile_max_labeled_demos', parseInt(e.target.value))}
                              className="bg-deep-black border border-zinc-800 text-zinc-100 font-mono rounded-lg p-3"
                              min="0"
                            />
                          </div>
                          <div>
                            <label className="block text-sm font-medium text-zinc-400 mb-1">
                              MiProv2 Compile Num Trials
                            </label>
                            <input
                              type="number"
                              value={config.miprov2_compile_num_trials}
                              onChange={(e) => handleConfigChange('miprov2_compile_num_trials', parseInt(e.target.value))}
                              className="bg-deep-black border border-zinc-800 text-zinc-100 font-mono rounded-lg p-3"
                              min="1"
                            />
                          </div>
                          <div>
                            <label className="block text-sm font-medium text-zinc-400 mb-1">
                              MiProv2 Compile Minibatch Size
                            </label>
                            <input
                              type="number"
                              value={config.miprov2_compile_minibatch_size}
                              onChange={(e) => handleConfigChange('miprov2_compile_minibatch_size', parseInt(e.target.value))}
                              className="bg-deep-black border border-zinc-800 text-zinc-100 font-mono rounded-lg p-3"
                              min="1"
                            />
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </PopoverContent>
              </Popover>

              <Popover open={isSessionOpen} onOpenChange={setIsSessionOpen}>
                <PopoverTrigger asChild>
                  <button className="btn btn-gray flex items-center gap-2">
                    <List className="h-5 w-5" />
                    Session
                  </button>
                </PopoverTrigger>
                <PopoverContent className="w-[300px] bg-[#232526] border border-zinc-800 shadow-lg p-6" align="end">
                  <div className="space-y-4">
                    <h3 className="text-lg font-medium text-zinc-200 mb-4">Session Management</h3>
                    <div className="space-y-3">
                      <Button
                        onClick={handleDownloadSession}
                        className="w-full h-10 bg-indigo-600 hover:bg-indigo-500 text-white"
                        disabled={!sessionData}
                      >
                        Download Session
                      </Button>
                      <div className="flex gap-2 items-center w-full">
                        <Button
                          onClick={() => document.getElementById('session-upload').click()}
                          className="w-full h-10 bg-indigo-600 hover:bg-indigo-500 text-white"
                        >
                          Upload Session
                        </Button>
                        <input
                          type="file"
                          id="session-upload"
                          className="hidden"
                          accept=".json"
                          onChange={(e) => handleUploadSession(e.target.files[0])}
                        />
                      </div>
                      <Button
                        onClick={() => {
                          startNewSession();
                          setSessionData(null);
                          setIsSessionOpen(false);
                        }}
                        className="w-full h-10 bg-zinc-700 hover:bg-zinc-600 text-white"
                      >
                        New Session
                      </Button>
                    </div>
                  </div>
                </PopoverContent>
              </Popover>
            </div>
          </div>

          {/* Navigation Tabs */}
          <div className="border-b border-zinc-800">
            <nav className="-mb-px flex space-x-8">
              <button
                onClick={() => setActiveTab('optimizer')}
                className={`${
                  activeTab === 'optimizer'
                    ? 'border-indigo-500 text-indigo-400'
                    : 'border-transparent text-zinc-400 hover:text-zinc-300 hover:border-zinc-700'
                } whitespace-nowrap pb-4 px-1 border-b-2 font-medium text-sm`}
              >
                Prompt Optimizer
              </button>
              <button
                onClick={() => setActiveTab('versions')}
                className={`${
                  activeTab === 'versions'
                    ? 'border-indigo-500 text-indigo-400'
                    : 'border-transparent text-zinc-400 hover:text-zinc-300 hover:border-zinc-700'
                } whitespace-nowrap pb-4 px-1 border-b-2 font-medium text-sm`}
              >
                Versions
              </button>
            </nav>
          </div>
        </div>

        {/* Main Content */}
        <div className="mt-6">
          {activeTab === 'optimizer' && (
            <PromptOptimizer 
              sessionData={sessionData} 
              onVersionUpdate={handleVersionUpdate}
              setSessionData={setSessionData}
              isConfigOpen={isConfigOpen}
              setIsConfigOpen={setIsConfigOpen}
              isSessionOpen={isSessionOpen}
              setIsSessionOpen={setIsSessionOpen}
              config={config}
              setConfig={setConfig}
            />
          )}
          {activeTab === 'versions' && <PromptVersions versions={versions} />}
        </div>
      </div>
    </div>
  );
}

export default App;