import { useState, useCallback } from 'react';
import { AnalysisResult, AnalysisData, AnalysisSettings, UploadProgress } from '../types';
import { api } from '../lib/api';
import { uploadAudioFile, saveAnalysisResult } from '../lib/supabase';
import { useAuth } from './useAuth';
import toast from 'react-hot-toast';

interface UseAnalysisReturn {
  isAnalyzing: boolean;
  progress: UploadProgress | null;
  analyzeFile: (file: File, settings: AnalysisSettings) => Promise<AnalysisResult>;
  cancelAnalysis: () => void;
  error: string | null;
}

export function useAnalysis(): UseAnalysisReturn {
  const { user } = useAuth();
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [progress, setProgress] = useState<UploadProgress | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [abortController, setAbortController] = useState<AbortController | null>(null);

  const analyzeFile = useCallback(async (
    file: File, 
    settings: AnalysisSettings
  ): Promise<AnalysisResult> => {
    if (!user) {
      throw new Error('User must be logged in to analyze files');
    }

    // Reset state
    setIsAnalyzing(true);
    setError(null);
    setProgress({
      progress: 0,
      stage: 'uploading',
      message: 'Preparing analysis...'
    });

    // Create abort controller for cancellation
    const controller = new AbortController();
    setAbortController(controller);

    try {
      // Validate file
      const validation = api.validateAudioFile(file);
      if (!validation.valid) {
        throw new Error(validation.error);
      }

      // Step 1: Upload file to Supabase storage
      setProgress({
        progress: 10,
        stage: 'uploading',
        message: 'Uploading file to secure storage...'
      });

      const fileUrl = await uploadAudioFile(file, user.id);

      // Step 2: Send to BeatWizard API for analysis
      setProgress({
        progress: 20,
        stage: 'processing',
        message: 'Sending to analysis engine...'
      });

      const analysisData = await api.analyzeAudioFile(file, settings, (progress) => {
        setProgress(progress);
      });

      // Step 3: Save results to database
      setProgress({
        progress: 95,
        stage: 'complete',
        message: 'Saving results...'
      });

      const analysisResult: Omit<AnalysisResult, 'id' | 'created_at'> = {
        user_id: user.id,
        file_name: file.name,
        file_url: fileUrl,
        analysis_data: analysisData,
        skill_level: settings.skill_level,
        genre: settings.genre || 'electronic'
      };

      const savedResult = await saveAnalysisResult(analysisResult);

      setProgress({
        progress: 100,
        stage: 'complete',
        message: 'Analysis complete!'
      });

      toast.success('Analysis completed successfully!');
      
      return savedResult;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Analysis failed';
      setError(errorMessage);
      toast.error(errorMessage);
      throw err;
    } finally {
      setIsAnalyzing(false);
      setAbortController(null);
      // Keep progress visible for a moment before clearing
      setTimeout(() => setProgress(null), 2000);
    }
  }, [user]);

  const cancelAnalysis = useCallback(() => {
    if (abortController) {
      abortController.abort();
      setAbortController(null);
    }
    setIsAnalyzing(false);
    setProgress(null);
    setError(null);
    toast.error('Analysis cancelled');
  }, [abortController]);

  return {
    isAnalyzing,
    progress,
    analyzeFile,
    cancelAnalysis,
    error
  };
}

// Hook for managing analysis history
export function useAnalysisHistory() {
  const { user } = useAuth();
  const [analyses, setAnalyses] = useState<AnalysisResult[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadAnalyses = useCallback(async (limit: number = 10) => {
    if (!user) return;

    try {
      setLoading(true);
      setError(null);
      
      const { data, error: supabaseError } = await supabase
        .from('analysis_results')
        .select('*')
        .eq('user_id', user.id)
        .order('created_at', { ascending: false })
        .limit(limit);

      if (supabaseError) throw supabaseError;
      
      setAnalyses(data || []);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to load analyses';
      setError(errorMessage);
      toast.error(errorMessage);
    } finally {
      setLoading(false);
    }
  }, [user]);

  const deleteAnalysis = useCallback(async (analysisId: string) => {
    try {
      const { error: supabaseError } = await supabase
        .from('analysis_results')
        .delete()
        .eq('id', analysisId);

      if (supabaseError) throw supabaseError;
      
      setAnalyses(prev => prev.filter(analysis => analysis.id !== analysisId));
      toast.success('Analysis deleted successfully');
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to delete analysis';
      toast.error(errorMessage);
      throw err;
    }
  }, []);

  // Load analyses when user changes
  useState(() => {
    loadAnalyses();
  }, [user, loadAnalyses]);

  return {
    analyses,
    loading,
    error,
    loadAnalyses,
    deleteAnalysis,
    refresh: () => loadAnalyses()
  };
}

// Hook for individual analysis
export function useAnalysisById(analysisId: string | null) {
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useState(async () => {
    if (!analysisId) {
      setLoading(false);
      return;
    }

    try {
      setLoading(true);
      setError(null);
      
      const { data, error: supabaseError } = await supabase
        .from('analysis_results')
        .select('*')
        .eq('id', analysisId)
        .single();

      if (supabaseError) throw supabaseError;
      
      setAnalysis(data);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to load analysis';
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  }, [analysisId]);

  return {
    analysis,
    loading,
    error
  };
}