import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Calendar, MapPin, Building, Eye, X } from "lucide-react";
import { cn } from "@/lib/utils";

interface Application {
  id: string;
  internshipTitle: string;
  company: string;
  location: string;
  appliedDate: string;
  status: 'pending' | 'accepted' | 'rejected' | 'withdrawn';
  matchScore: number;
  deadline?: string;
  response?: string;
}

interface ApplicationTrackerProps {
  applications: Application[];
  onViewDetails?: (applicationId: string) => void;
  onWithdraw?: (applicationId: string) => void;
  showActions?: boolean;
}

export default function ApplicationTracker({
  applications,
  onViewDetails,
  onWithdraw,
  showActions = true
}: ApplicationTrackerProps) {
  const getStatusConfig = (status: string) => {
    switch (status) {
      case 'accepted':
        return {
          color: "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200",
          progress: 100
        };
      case 'rejected':
        return {
          color: "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200",
          progress: 100
        };
      case 'withdrawn':
        return {
          color: "bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200",
          progress: 0
        };
      default:
        return {
          color: "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200",
          progress: 50
        };
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric'
    });
  };

  const getTimeRemaining = (deadline: string) => {
    const now = new Date();
    const deadlineDate = new Date(deadline);
    const diffTime = deadlineDate.getTime() - now.getTime();
    const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
    
    if (diffDays < 0) return "Expired";
    if (diffDays === 0) return "Today";
    if (diffDays === 1) return "1 day left";
    return `${diffDays} days left`;
  };

  return (
    <Card data-testid="card-application-tracker">
      <CardHeader>
        <CardTitle className="text-lg font-semibold">
          My Applications ({applications.length})
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {applications.map((application) => {
            const statusConfig = getStatusConfig(application.status);
            const timeRemaining = application.deadline ? getTimeRemaining(application.deadline) : null;
            
            return (
              <div
                key={application.id}
                className="border border-border rounded-lg p-4 hover-elevate transition-all duration-200"
                data-testid={`application-${application.id}`}
              >
                <div className="flex items-start justify-between mb-3">
                  <div className="flex-1">
                    <h4 className="font-medium text-base leading-tight" data-testid="text-internship-title">
                      {application.internshipTitle}
                    </h4>
                    <div className="flex items-center gap-4 mt-1 text-sm text-muted-foreground">
                      <div className="flex items-center gap-1">
                        <Building className="h-3 w-3" />
                        <span>{application.company}</span>
                      </div>
                      <div className="flex items-center gap-1">
                        <MapPin className="h-3 w-3" />
                        <span>{application.location}</span>
                      </div>
                    </div>
                  </div>
                  
                  <div className="flex items-center gap-2">
                    <div className="text-right">
                      <div className="text-lg font-semibold text-primary" data-testid="text-match-score">
                        {application.matchScore}%
                      </div>
                      <div className="text-xs text-muted-foreground">Match</div>
                    </div>
                  </div>
                </div>

                <Progress value={statusConfig.progress} className="mb-3" data-testid="progress-application" />

                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    <Badge className={statusConfig.color} data-testid="badge-status">
                      {application.status.charAt(0).toUpperCase() + application.status.slice(1)}
                    </Badge>
                    
                    <div className="flex items-center gap-1 text-xs text-muted-foreground">
                      <Calendar className="h-3 w-3" />
                      <span>Applied {formatDate(application.appliedDate)}</span>
                    </div>

                    {timeRemaining && application.status === 'pending' && (
                      <div className={cn(
                        "text-xs font-medium",
                        timeRemaining.includes('day') ? "text-orange-600" : "text-red-600"
                      )}>
                        {timeRemaining}
                      </div>
                    )}
                  </div>

                  {showActions && (
                    <div className="flex items-center gap-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => onViewDetails?.(application.id)}
                        data-testid="button-view-details"
                      >
                        <Eye className="h-4 w-4 mr-1" />
                        View
                      </Button>
                      
                      {application.status === 'pending' && (
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => onWithdraw?.(application.id)}
                          data-testid="button-withdraw"
                        >
                          <X className="h-4 w-4 mr-1" />
                          Withdraw
                        </Button>
                      )}
                    </div>
                  )}
                </div>

                {application.response && (
                  <div className="mt-3 p-3 bg-muted rounded-md">
                    <p className="text-sm">{application.response}</p>
                  </div>
                )}
              </div>
            );
          })}

          {applications.length === 0 && (
            <div className="text-center py-8 text-muted-foreground">
              <p>No applications yet</p>
              <p className="text-sm">Start applying to internships to see your progress here</p>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}