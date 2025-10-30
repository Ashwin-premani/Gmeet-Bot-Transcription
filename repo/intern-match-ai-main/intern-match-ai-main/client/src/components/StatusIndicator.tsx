import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { cn } from "@/lib/utils";

interface StatusIndicatorProps {
  status: 'pending' | 'accepted' | 'rejected' | 'open' | 'closed' | 'filled';
  progress?: number;
  showProgress?: boolean;
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

export default function StatusIndicator({ 
  status, 
  progress = 0, 
  showProgress = false,
  size = 'md',
  className 
}: StatusIndicatorProps) {
  const getStatusConfig = (status: string) => {
    switch (status) {
      case 'accepted':
        return {
          color: "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200",
          label: "Accepted",
          dot: "bg-green-500"
        };
      case 'rejected':
        return {
          color: "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200",
          label: "Rejected",
          dot: "bg-red-500"
        };
      case 'pending':
        return {
          color: "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200",
          label: "Pending",
          dot: "bg-yellow-500"
        };
      case 'open':
        return {
          color: "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200",
          label: "Open",
          dot: "bg-blue-500"
        };
      case 'closed':
        return {
          color: "bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200",
          label: "Closed",
          dot: "bg-gray-500"
        };
      case 'filled':
        return {
          color: "bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200",
          label: "Filled",
          dot: "bg-purple-500"
        };
      default:
        return {
          color: "bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200",
          label: status,
          dot: "bg-gray-500"
        };
    }
  };

  const config = getStatusConfig(status);
  
  const sizeClasses = {
    sm: "text-xs px-2 py-1",
    md: "text-sm px-3 py-1",
    lg: "text-base px-4 py-2"
  };

  const dotSizes = {
    sm: "w-2 h-2",
    md: "w-3 h-3", 
    lg: "w-4 h-4"
  };

  return (
    <div className={cn("flex flex-col gap-2", className)}>
      <div className="flex items-center gap-2">
        <div className={cn("rounded-full", config.dot, dotSizes[size])} />
        <Badge 
          className={cn(config.color, sizeClasses[size])}
          data-testid={`status-${status}`}
        >
          {config.label}
        </Badge>
      </div>
      
      {showProgress && (
        <div className="space-y-1">
          <Progress value={progress} className="h-2" data-testid="progress-status" />
          <div className="text-xs text-muted-foreground text-right">
            {progress}% complete
          </div>
        </div>
      )}
    </div>
  );
}