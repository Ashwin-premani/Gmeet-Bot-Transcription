import { Card, CardContent, CardFooter, CardHeader } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { HelpCircle, MapPin, Calendar, DollarSign, Building, Eye, ChevronDown, Brain, Target, AlertTriangle, Lightbulb, TrendingUp } from "lucide-react";
import { useLocation } from "wouter";
import { useState } from "react";

interface AIAnalysis {
  overallMatch: number;
  confidence: number;
  keyStrengths: string[];
  potentialConcerns: string[];
  skillGaps: string[];
  careerImpact: string;
  employerBenefits: string[];
  actionableAdvice: string[];
  breakdown: {
    skillsMatch: number;
    experienceMatch: number;
    locationMatch: number;
    cultureMatch: number;
    careerFitMatch: number;
  };
}

interface MatchCardProps {
  id: string;
  title: string;
  company?: string;
  location: string;
  matchScore: number;
  matchReasons: string[];
  duration?: string;
  stipend?: string;
  skills: string[];
  status: 'pending' | 'accepted' | 'rejected' | 'open' | 'closed';
  onApply?: () => void;
  onAccept?: () => void;
  onReject?: () => void;
  showActions?: boolean;
  type: 'internship' | 'candidate';
  candidateName?: string;
  university?: string;
  aiAnalysis?: AIAnalysis;
}

export default function MatchCard({
  id,
  title,
  company,
  location,
  matchScore,
  matchReasons,
  duration,
  stipend,
  skills,
  status,
  onApply,
  onAccept,
  onReject,
  showActions = true,
  type,
  candidateName,
  university,
  aiAnalysis
}: MatchCardProps) {
  const [, setLocation] = useLocation();
  const [showAIDetails, setShowAIDetails] = useState(false);
  const getScoreColor = (score: number) => {
    if (score >= 80) return "text-green-600";
    if (score >= 60) return "text-yellow-600";
    return "text-red-600";
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'accepted': return "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200";
      case 'rejected': return "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200";
      case 'pending': return "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200";
      default: return "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200";
    }
  };

  const handleViewDetails = () => {
    if (type === 'internship') {
      setLocation(`/internship/${id}`);
    } else {
      setLocation(`/student/${id}`);
    }
  };

  return (
    <Card className="hover-elevate transition-all duration-200" data-testid={`card-${type}-${title.replace(/\s+/g, '-').toLowerCase()}`}>
      <CardHeader className="pb-4">
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <h3 className="font-semibold text-lg leading-tight" data-testid={`text-${type}-title`}>
              {type === 'candidate' ? candidateName : title}
            </h3>
            <p className="text-muted-foreground text-sm mt-1">
              {type === 'candidate' ? university : company}
            </p>
            {type === 'internship' && (
              <p className="text-sm font-medium text-foreground mt-1">{title}</p>
            )}
          </div>
          <div className="flex items-center gap-2">
            <div className="text-right">
              <div className={`text-2xl font-bold ${getScoreColor(matchScore)}`} data-testid="text-match-score">
                {matchScore}%
              </div>
              <div className="text-xs text-muted-foreground">AI Match</div>
            </div>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button variant="ghost" size="icon" className="h-6 w-6" data-testid="button-match-info">
                  <HelpCircle className="h-4 w-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent className="max-w-xs">
                <div className="space-y-1">
                  <p className="font-medium">Why this match?</p>
                  <ul className="text-xs space-y-1">
                    {matchReasons.map((reason, index) => (
                      <li key={index}>• {reason}</li>
                    ))}
                  </ul>
                </div>
              </TooltipContent>
            </Tooltip>
          </div>
        </div>
        <Progress value={matchScore} className="mt-3" data-testid="progress-match-score" />
      </CardHeader>

      <CardContent className="space-y-4">
        <div className="flex items-center gap-4 text-sm text-muted-foreground">
          <div className="flex items-center gap-1">
            <MapPin className="h-4 w-4" />
            <span>{location}</span>
          </div>
          {duration && (
            <div className="flex items-center gap-1">
              <Calendar className="h-4 w-4" />
              <span>{duration}</span>
            </div>
          )}
          {stipend && (
            <div className="flex items-center gap-1">
              <DollarSign className="h-4 w-4" />
              <span>{stipend}</span>
            </div>
          )}
        </div>

        <div className="flex flex-wrap gap-2">
          {skills.slice(0, 6).map((skill, index) => (
            <Badge key={index} variant="secondary" className="text-xs" data-testid={`badge-skill-${index}`}>
              {skill}
            </Badge>
          ))}
          {skills.length > 6 && (
            <Badge variant="outline" className="text-xs">
              +{skills.length - 6} more
            </Badge>
          )}
        </div>

        <Badge className={getStatusColor(status)} data-testid="badge-status">
          {status.charAt(0).toUpperCase() + status.slice(1)}
        </Badge>

        {/* AI Insights Section */}
        {aiAnalysis && (
          <Collapsible open={showAIDetails} onOpenChange={setShowAIDetails}>
            <CollapsibleTrigger asChild>
              <Button variant="ghost" className="w-full justify-between p-2 h-auto">
                <div className="flex items-center gap-2">
                  <Brain className="h-4 w-4 text-blue-600" />
                  <span className="text-sm font-medium">AI Insights</span>
                  <Badge variant="outline" className="text-xs">
                    {aiAnalysis.confidence}% confidence
                  </Badge>
                </div>
                <ChevronDown className={`h-4 w-4 transition-transform ${showAIDetails ? 'rotate-180' : ''}`} />
              </Button>
            </CollapsibleTrigger>
            
            <CollapsibleContent className="space-y-3 pt-3">
              {/* Match Breakdown */}
              <div className="space-y-2">
                <h4 className="text-xs font-medium text-muted-foreground uppercase tracking-wide">Match Breakdown</h4>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div className="flex justify-between">
                    <span>Skills:</span>
                    <span className={getScoreColor(aiAnalysis.breakdown.skillsMatch)}>{aiAnalysis.breakdown.skillsMatch}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Experience:</span>
                    <span className={getScoreColor(aiAnalysis.breakdown.experienceMatch)}>{aiAnalysis.breakdown.experienceMatch}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Location:</span>
                    <span className={getScoreColor(aiAnalysis.breakdown.locationMatch)}>{aiAnalysis.breakdown.locationMatch}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Culture:</span>
                    <span className={getScoreColor(aiAnalysis.breakdown.cultureMatch)}>{aiAnalysis.breakdown.cultureMatch}%</span>
                  </div>
                </div>
              </div>

              {/* Key Strengths */}
              {aiAnalysis.keyStrengths.length > 0 && (
                <div className="space-y-2">
                  <div className="flex items-center gap-1">
                    <Target className="h-3 w-3 text-green-600" />
                    <h4 className="text-xs font-medium text-green-600">Key Strengths</h4>
                  </div>
                  <ul className="text-xs space-y-1 text-muted-foreground">
                    {aiAnalysis.keyStrengths.slice(0, 3).map((strength, index) => (
                      <li key={index} className="flex items-start gap-1">
                        <span className="text-green-600 mt-0.5">•</span>
                        <span>{strength}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Concerns */}
              {aiAnalysis.potentialConcerns.length > 0 && (
                <div className="space-y-2">
                  <div className="flex items-center gap-1">
                    <AlertTriangle className="h-3 w-3 text-yellow-600" />
                    <h4 className="text-xs font-medium text-yellow-600">Considerations</h4>
                  </div>
                  <ul className="text-xs space-y-1 text-muted-foreground">
                    {aiAnalysis.potentialConcerns.slice(0, 2).map((concern, index) => (
                      <li key={index} className="flex items-start gap-1">
                        <span className="text-yellow-600 mt-0.5">•</span>
                        <span>{concern}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Actionable Advice */}
              {aiAnalysis.actionableAdvice.length > 0 && (
                <div className="space-y-2">
                  <div className="flex items-center gap-1">
                    <Lightbulb className="h-3 w-3 text-blue-600" />
                    <h4 className="text-xs font-medium text-blue-600">Recommendations</h4>
                  </div>
                  <ul className="text-xs space-y-1 text-muted-foreground">
                    {aiAnalysis.actionableAdvice.slice(0, 2).map((advice, index) => (
                      <li key={index} className="flex items-start gap-1">
                        <span className="text-blue-600 mt-0.5">•</span>
                        <span>{advice}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Career Impact */}
              {aiAnalysis.careerImpact && (
                <div className="space-y-2">
                  <div className="flex items-center gap-1">
                    <TrendingUp className="h-3 w-3 text-purple-600" />
                    <h4 className="text-xs font-medium text-purple-600">Career Impact</h4>
                  </div>
                  <p className="text-xs text-muted-foreground leading-relaxed">
                    {aiAnalysis.careerImpact}
                  </p>
                </div>
              )}
            </CollapsibleContent>
          </Collapsible>
        )}
      </CardContent>

      {showActions && (
        <CardFooter className="pt-4 gap-2">
          <Button 
            onClick={handleViewDetails} 
            variant="outline"
            className="flex items-center gap-2"
            data-testid="button-view-details"
          >
            <Eye className="h-4 w-4" />
            View Details
          </Button>
          
          {type === 'internship' && status === 'open' && (
            <Button 
              onClick={onApply} 
              className="flex-1"
              data-testid="button-apply"
            >
              Apply Now
            </Button>
          )}
          {type === 'candidate' && status === 'pending' && (
            <>
              <Button 
                onClick={onAccept} 
                variant="default" 
                className="flex-1"
                data-testid="button-accept"
              >
                Accept
              </Button>
              <Button 
                onClick={onReject} 
                variant="outline" 
                className="flex-1"
                data-testid="button-reject"
              >
                Reject
              </Button>
            </>
          )}
        </CardFooter>
      )}
    </Card>
  );
}