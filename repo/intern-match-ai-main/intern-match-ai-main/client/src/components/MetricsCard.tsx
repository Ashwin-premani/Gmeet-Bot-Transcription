import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { TrendingUp, TrendingDown, Minus } from "lucide-react";
import { cn } from "@/lib/utils";

interface MetricsCardProps {
  title: string;
  value: string | number;
  description?: string;
  trend?: {
    value: number;
    label: string;
    direction: 'up' | 'down' | 'neutral';
  };
  progress?: {
    value: number;
    max: number;
    label?: string;
  };
  icon?: React.ReactNode;
  className?: string;
}

export default function MetricsCard({
  title,
  value,
  description,
  trend,
  progress,
  icon,
  className
}: MetricsCardProps) {
  const getTrendIcon = (direction: string) => {
    switch (direction) {
      case 'up':
        return <TrendingUp className="h-4 w-4 text-green-500" />;
      case 'down':
        return <TrendingDown className="h-4 w-4 text-red-500" />;
      default:
        return <Minus className="h-4 w-4 text-gray-500" />;
    }
  };

  const getTrendColor = (direction: string) => {
    switch (direction) {
      case 'up':
        return "text-green-600";
      case 'down':
        return "text-red-600";
      default:
        return "text-gray-600";
    }
  };

  const progressPercentage = progress ? (progress.value / progress.max) * 100 : 0;

  return (
    <Card className={cn("", className)} data-testid={`card-metrics-${title.replace(/\s+/g, '-').toLowerCase()}`}>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium text-muted-foreground">
          {title}
        </CardTitle>
        {icon && (
          <div className="text-muted-foreground">
            {icon}
          </div>
        )}
      </CardHeader>
      <CardContent>
        <div className="text-3xl font-bold" data-testid="text-metric-value">
          {value}
        </div>
        
        {description && (
          <p className="text-xs text-muted-foreground mt-1">
            {description}
          </p>
        )}

        {trend && (
          <div className="flex items-center mt-2">
            {getTrendIcon(trend.direction)}
            <span className={cn("text-xs ml-1", getTrendColor(trend.direction))}>
              {trend.value > 0 ? '+' : ''}{trend.value}% {trend.label}
            </span>
          </div>
        )}

        {progress && (
          <div className="mt-3 space-y-2">
            <Progress value={progressPercentage} className="h-2" data-testid="progress-metric" />
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>{progress.label || 'Progress'}</span>
              <span>{progress.value}/{progress.max}</span>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}