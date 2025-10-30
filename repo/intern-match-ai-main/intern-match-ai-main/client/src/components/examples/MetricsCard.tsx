import MetricsCard from '../MetricsCard';
import { Users, Briefcase, TrendingUp, Target } from 'lucide-react';

export default function MetricsCardExample() {
  return (
    <div className="p-6 space-y-6 bg-background">
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <MetricsCard
          title="Total Applications"
          value="1,234"
          description="Active applications"
          trend={{
            value: 12,
            label: "from last month",
            direction: "up"
          }}
          icon={<Briefcase className="h-4 w-4" />}
        />

        <MetricsCard
          title="Successful Matches"
          value="87%"
          description="Match success rate"
          trend={{
            value: -3,
            label: "from last month", 
            direction: "down"
          }}
          icon={<Target className="h-4 w-4" />}
        />

        <MetricsCard
          title="Active Students"
          value="2,456"
          description="Registered students"
          progress={{
            value: 456,
            max: 500,
            label: "Capacity"
          }}
          icon={<Users className="h-4 w-4" />}
        />

        <MetricsCard
          title="Average Match Score"
          value="84.2"
          description="AI matching accuracy"
          trend={{
            value: 0,
            label: "no change",
            direction: "neutral"
          }}
          icon={<TrendingUp className="h-4 w-4" />}
        />
      </div>
    </div>
  );
}