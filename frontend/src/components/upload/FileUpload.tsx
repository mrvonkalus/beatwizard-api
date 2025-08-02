import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, X, Music, AlertCircle } from 'lucide-react';
import { Button } from '../ui/Button';
import { Card, CardContent } from '../ui/Card';
import { Progress, StepProgress } from '../ui/Progress';
import { Select } from '../ui/Input';
import { AnalysisSettings, UploadProgress } from '../../types';
import { useAnalysis } from '../../hooks/useAnalysis';
import { api } from '../../lib/api';
import { formatFileSize } from '../../utils';

interface FileUploadProps {
  onAnalysisComplete: (analysisId: string) => void;
  onAnalysisStart?: () => void;
}

export function FileUpload({ onAnalysisComplete, onAnalysisStart }: FileUploadProps) {
  const { isAnalyzing, progress, analyzeFile, cancelAnalysis, error } = useAnalysis();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [settings, setSettings] = useState<AnalysisSettings>({
    skill_level: 'beginner',
    genre: 'electronic',
    include_advanced_features: true,
    focus_areas: []
  });

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (file) {
      const validation = api.validateAudioFile(file);
      if (validation.valid) {
        setSelectedFile(file);
      } else {
        alert(validation.error);
      }
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    accept: {
      'audio/*': ['.mp3', '.wav', '.flac', '.m4a', '.aiff']
    },
    maxFiles: 1,
    maxSize: 100 * 1024 * 1024, // 100MB
    disabled: isAnalyzing
  });

  const handleAnalyze = async () => {
    if (!selectedFile) return;

    try {
      onAnalysisStart?.();
      const result = await analyzeFile(selectedFile, settings);
      onAnalysisComplete(result.id);
    } catch (err) {
      console.error('Analysis failed:', err);
    }
  };

  const removeFile = () => {
    setSelectedFile(null);
  };

  const getProgressSteps = () => {
    if (!progress) return [];
    
    return [
      {
        label: 'Upload',
        description: 'Uploading your audio file',
        completed: progress.stage !== 'uploading'
      },
      {
        label: 'Process',
        description: 'Processing audio content',
        completed: ['analyzing', 'generating_feedback', 'complete'].includes(progress.stage)
      },
      {
        label: 'Analyze',
        description: 'Running professional analysis',
        completed: ['generating_feedback', 'complete'].includes(progress.stage)
      },
      {
        label: 'Feedback',
        description: 'Generating intelligent feedback',
        completed: progress.stage === 'complete'
      }
    ];
  };

  if (isAnalyzing && progress) {
    return (
      <Card className="w-full max-w-2xl mx-auto">
        <CardContent className="text-center space-y-6">
          <div className="flex items-center justify-center w-20 h-20 mx-auto bg-purple-100 rounded-full">
            <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-purple-600"></div>
          </div>
          
          <div>
            <h3 className="text-xl font-semibold text-gray-900 mb-2">
              Analyzing Your Track
            </h3>
            <p className="text-gray-600 mb-4">
              {progress.message}
            </p>
            
            <Progress
              value={progress.progress}
              size="lg"
              color="purple"
              animated
              showLabel
              className="mb-6"
            />
            
            <StepProgress
              currentStep={getProgressSteps().filter(s => s.completed).length + 1}
              totalSteps={4}
              steps={[
                { label: 'Upload', description: 'Secure file upload' },
                { label: 'Process', description: 'Audio preprocessing' },
                { label: 'Analyze', description: 'Professional analysis' },
                { label: 'Feedback', description: 'AI-powered insights' }
              ]}
              variant="horizontal"
            />
          </div>
          
          <Button
            variant="outline"
            onClick={cancelAnalysis}
            className="mt-4"
          >
            Cancel Analysis
          </Button>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="w-full max-w-2xl mx-auto space-y-6">
      {/* File Drop Zone */}
      <Card>
        <CardContent>
          <div
            {...getRootProps()}
            className={`
              border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all duration-200
              ${isDragActive && !isDragReject ? 'border-purple-400 bg-purple-50' : ''}
              ${isDragReject ? 'border-red-400 bg-red-50' : ''}
              ${!isDragActive ? 'border-gray-300 hover:border-purple-400 hover:bg-gray-50' : ''}
              ${selectedFile ? 'border-green-400 bg-green-50' : ''}
              ${isAnalyzing ? 'opacity-50 cursor-not-allowed' : ''}
            `}
          >
            <input {...getInputProps()} />
            
            <div className="space-y-4">
              {selectedFile ? (
                <>
                  <div className="flex items-center justify-center w-16 h-16 mx-auto bg-green-100 rounded-full">
                    <Music className="w-8 h-8 text-green-600" />
                  </div>
                  <div>
                    <p className="text-lg font-medium text-gray-900">
                      {selectedFile.name}
                    </p>
                    <p className="text-sm text-gray-500">
                      {formatFileSize(selectedFile.size)} • {selectedFile.type}
                    </p>
                  </div>
                  <Button
                    variant="ghost"
                    onClick={(e) => {
                      e.stopPropagation();
                      removeFile();
                    }}
                    icon={<X size={16} />}
                  >
                    Remove file
                  </Button>
                </>
              ) : (
                <>
                  <div className="flex items-center justify-center w-16 h-16 mx-auto bg-purple-100 rounded-full">
                    <Upload className="w-8 h-8 text-purple-600" />
                  </div>
                  <div>
                    <p className="text-lg font-medium text-gray-900">
                      {isDragActive
                        ? isDragReject
                          ? 'File type not supported'
                          : 'Drop your audio file here'
                        : 'Drag & drop your audio file here'
                      }
                    </p>
                    <p className="text-sm text-gray-500 mt-1">
                      or click to browse
                    </p>
                  </div>
                  <div className="text-xs text-gray-400">
                    Supports MP3, WAV, FLAC, M4A • Max 100MB
                  </div>
                </>
              )}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Analysis Settings */}
      {selectedFile && (
        <Card>
          <CardContent>
            <h3 className="text-lg font-semibold text-gray-900 mb-4">
              Analysis Settings
            </h3>
            
            <div className="grid md:grid-cols-2 gap-4">
              <Select
                label="Your Skill Level"
                value={settings.skill_level}
                onChange={(e) => setSettings({
                  ...settings,
                  skill_level: e.target.value as any
                })}
                options={api.getSkillLevelOptions().map(option => ({
                  value: option.value,
                  label: option.label
                }))}
              />
              
              <Select
                label="Genre"
                value={settings.genre}
                onChange={(e) => setSettings({
                  ...settings,
                  genre: e.target.value
                })}
                options={api.getGenreOptions()}
              />
            </div>
            
            <div className="mt-4">
              <label className="flex items-start">
                <input
                  type="checkbox"
                  checked={settings.include_advanced_features}
                  onChange={(e) => setSettings({
                    ...settings,
                    include_advanced_features: e.target.checked
                  })}
                  className="mt-1 mr-3 h-4 w-4 text-purple-600 border-gray-300 rounded focus:ring-purple-500"
                />
                <div>
                  <span className="text-sm font-medium text-gray-900">
                    Include Advanced Features
                  </span>
                  <p className="text-xs text-gray-500 mt-1">
                    Sound selection analysis, rhythmic patterns, harmonic progression, and professional mixing insights
                  </p>
                </div>
              </label>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Error Display */}
      {error && (
        <Card className="border-red-200 bg-red-50">
          <CardContent>
            <div className="flex items-start">
              <AlertCircle className="w-5 h-5 text-red-600 mt-0.5 mr-3 flex-shrink-0" />
              <div>
                <p className="text-sm font-medium text-red-800">
                  Analysis Failed
                </p>
                <p className="text-sm text-red-700 mt-1">
                  {error}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Analyze Button */}
      {selectedFile && (
        <div className="text-center">
          <Button
            variant="primary"
            size="lg"
            onClick={handleAnalyze}
            disabled={isAnalyzing}
            className="w-full md:w-auto"
          >
            Start Professional Analysis
          </Button>
          <p className="text-xs text-gray-500 mt-2">
            Analysis typically takes 60-90 seconds
          </p>
        </div>
      )}
    </div>
  );
}