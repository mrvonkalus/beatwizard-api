import React from 'react';
import { Link } from 'react-router-dom';
import { Music, Zap, Target, Users, ArrowRight, Play } from 'lucide-react';
import { Button } from '../components/ui/Button';
import { Card, CardContent } from '../components/ui/Card';

export function HomePage() {
  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-purple-600/20 to-pink-600/20"></div>
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-24">
          <div className="text-center">
            <h1 className="text-4xl md:text-6xl font-bold text-gray-900 mb-6">
              Get <span className="bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">
                Professional Feedback
              </span><br />
              on Your Tracks
            </h1>
            
            <p className="text-xl text-gray-600 max-w-3xl mx-auto mb-8">
              Stop guessing what's wrong with your mix. BeatWizard gives you specific, actionable feedback 
              just like a professional producer would - powered by advanced AI and professional audio analysis.
            </p>
            
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link to="/login">
                <Button variant="primary" size="lg" icon={<Music />}>
                  Start Analyzing
                </Button>
              </Link>
              <Button variant="outline" size="lg" icon={<Play />}>
                Watch Demo
              </Button>
            </div>
            
            <p className="text-sm text-gray-500 mt-4">
              ✨ Free analysis • No credit card required • Results in 60 seconds
            </p>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 bg-white/50 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">
              Why Producers Choose BeatWizard
            </h2>
            <p className="text-xl text-gray-600">
              Professional-grade analysis that actually helps you improve
            </p>
          </div>
          
          <div className="grid md:grid-cols-3 gap-8">
            <Card className="text-center p-8 hover:shadow-lg transition-shadow">
              <CardContent>
                <div className="w-16 h-16 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full flex items-center justify-center mx-auto mb-6">
                  <Target className="w-8 h-8 text-white" />
                </div>
                <h3 className="text-xl font-semibold mb-4">Specific Feedback</h3>
                <p className="text-gray-600">
                  "Your kick needs more punch around 80Hz" - not generic advice, actual solutions you can use immediately.
                </p>
              </CardContent>
            </Card>
            
            <Card className="text-center p-8 hover:shadow-lg transition-shadow">
              <CardContent>
                <div className="w-16 h-16 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full flex items-center justify-center mx-auto mb-6">
                  <Zap className="w-8 h-8 text-white" />
                </div>
                <h3 className="text-xl font-semibold mb-4">AI-Powered Insights</h3>
                <p className="text-gray-600">
                  Smart feedback that adapts to your skill level and genre, from beginner tips to pro techniques.
                </p>
              </CardContent>
            </Card>
            
            <Card className="text-center p-8 hover:shadow-lg transition-shadow">
              <CardContent>
                <div className="w-16 h-16 bg-gradient-to-r from-green-500 to-blue-500 rounded-full flex items-center justify-center mx-auto mb-6">
                  <Users className="w-8 h-8 text-white" />
                </div>
                <h3 className="text-xl font-semibold mb-4">Track Your Progress</h3>
                <p className="text-gray-600">
                  See your mixing skills improve over time with detailed progress tracking and skill assessments.
                </p>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* Analysis Preview */}
      <section className="py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">
              See What BeatWizard Finds
            </h2>
            <p className="text-xl text-gray-600">
              Professional analysis that identifies exactly what needs improvement
            </p>
          </div>
          
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            <div className="space-y-6">
              <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                <div className="flex items-start">
                  <div className="w-2 h-2 bg-red-500 rounded-full mt-2 mr-3"></div>
                  <div>
                    <p className="font-semibold text-red-800">🥁 Kick: Lacks punch and clarity</p>
                    <p className="text-red-600 text-sm mt-1">Try a kick with more 60-80Hz content</p>
                  </div>
                </div>
              </div>
              
              <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                <div className="flex items-start">
                  <div className="w-2 h-2 bg-yellow-500 rounded-full mt-2 mr-3"></div>
                  <div>
                    <p className="font-semibold text-yellow-800">🔊 Loudness: Too quiet for streaming</p>
                    <p className="text-yellow-600 text-sm mt-1">Increase level by +16dB for Spotify compliance</p>
                  </div>
                </div>
              </div>
              
              <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                <div className="flex items-start">
                  <div className="w-2 h-2 bg-green-500 rounded-full mt-2 mr-3"></div>
                  <div>
                    <p className="font-semibold text-green-800">📦 Sample Packs</p>
                    <p className="text-green-600 text-sm mt-1">KSHMR Kick Collection, Modern Trap Kicks Vol. 3</p>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="bg-gradient-to-br from-purple-600 to-pink-600 rounded-2xl p-8 text-white">
              <h3 className="text-2xl font-bold mb-4">Professional Analysis Includes:</h3>
              <ul className="space-y-3">
                <li className="flex items-center">
                  <div className="w-2 h-2 bg-white rounded-full mr-3"></div>
                  Tempo & Key Detection
                </li>
                <li className="flex items-center">
                  <div className="w-2 h-2 bg-white rounded-full mr-3"></div>
                  7-Band Frequency Analysis
                </li>
                <li className="flex items-center">
                  <div className="w-2 h-2 bg-white rounded-full mr-3"></div>
                  Professional LUFS Measurement
                </li>
                <li className="flex items-center">
                  <div className="w-2 h-2 bg-white rounded-full mr-3"></div>
                  Stereo Imaging Analysis
                </li>
                <li className="flex items-center">
                  <div className="w-2 h-2 bg-white rounded-full mr-3"></div>
                  Sound Selection Feedback
                </li>
                <li className="flex items-center">
                  <div className="w-2 h-2 bg-white rounded-full mr-3"></div>
                  AI-Powered Recommendations
                </li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 bg-gradient-to-r from-purple-600 to-pink-600">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="text-3xl font-bold text-white mb-4">
            Ready to Level Up Your Tracks?
          </h2>
          <p className="text-xl text-purple-100 mb-8">
            Join thousands of producers getting professional feedback on their music
          </p>
          
          <Link to="/login">
            <Button variant="secondary" size="lg" icon={<ArrowRight />}>
              Start Your Free Analysis
            </Button>
          </Link>
          
          <div className="flex justify-center items-center space-x-8 mt-8 text-purple-100">
            <div className="flex items-center">
              <Zap className="w-5 h-5 mr-2" />
              <span>60 second analysis</span>
            </div>
            <div className="flex items-center">
              <Users className="w-5 h-5 mr-2" />
              <span>Professional quality</span>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}