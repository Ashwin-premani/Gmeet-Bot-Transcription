import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Progress } from "@/components/ui/progress";
import MetricsCard from "./MetricsCard";
import QuotaChart from "./QuotaChart";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line, PieChart, Pie, Cell } from "recharts";
import { TrendingUp, Users, Briefcase, Target, AlertTriangle, Download, Filter, Calendar } from "lucide-react";

interface AdminMetrics {
  totalStudents: number;
  totalEmployers: number;
  totalInternships: number;
  totalApplications: number;
  successfulMatches: number;
  averageMatchScore: number;
  quotaCompliance: number;
  fairnessScore: number;
}

interface QuotaData {
  category: string;
  allocated: number;
  total: number;
  percentage: number;
  color: string;
}

interface CapacityData {
  month: string;
  students: number;
  internships: number;
  applications: number;
}

interface FairnessAlert {
  id: string;
  type: 'quota' | 'bias' | 'capacity';
  severity: 'high' | 'medium' | 'low';
  message: string;
  timestamp: string;
}

interface AdminDashboardProps {
  metrics: AdminMetrics;
  quotaData: QuotaData[];
  capacityData: CapacityData[];
  fairnessAlerts: FairnessAlert[];
  onExportData?: (type: string) => void;
  onViewDetails?: (type: string) => void;
}

export default function AdminDashboard({
  metrics,
  quotaData,
  capacityData,
  fairnessAlerts,
  onExportData,
  onViewDetails
}: AdminDashboardProps) {
  const matchSuccessRate = metrics.totalApplications > 0 
    ? ((metrics.successfulMatches / metrics.totalApplications) * 100).toFixed(1)
    : 0;

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'high': return "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200";
      case 'medium': return "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200";
      default: return "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200";
    }
  };

  return (
    <div className="space-y-6 p-6" data-testid="dashboard-admin">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Admin Dashboard</h1>
          <p className="text-muted-foreground">System overview and compliance monitoring</p>
        </div>
        <div className="flex items-center gap-2">
          <Select defaultValue="all">
            <SelectTrigger className="w-32" data-testid="select-time-period">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Time</SelectItem>
              <SelectItem value="month">This Month</SelectItem>
              <SelectItem value="quarter">This Quarter</SelectItem>
              <SelectItem value="year">This Year</SelectItem>
            </SelectContent>
          </Select>
          <Button variant="outline" onClick={() => onExportData?.('full')} data-testid="button-export-data">
            <Download className="h-4 w-4 mr-2" />
            Export Data
          </Button>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <MetricsCard
          title="Total Students"
          value={metrics.totalStudents.toLocaleString()}
          description="Registered users"
          trend={{ value: 12, label: "from last month", direction: "up" }}
          icon={<Users className="h-4 w-4" />}
        />
        <MetricsCard
          title="Active Internships"
          value={metrics.totalInternships}
          description="Currently available"
          trend={{ value: 8, label: "from last month", direction: "up" }}
          icon={<Briefcase className="h-4 w-4" />}
        />
        <MetricsCard
          title="Match Success Rate"
          value={`${matchSuccessRate}%`}
          description="Successful placements"
          trend={{ value: 3, label: "from last month", direction: "up" }}
          icon={<Target className="h-4 w-4" />}
        />
        <MetricsCard
          title="Fairness Score"
          value={`${metrics.fairnessScore}/100`}
          description="Bias monitoring"
          progress={{ value: metrics.fairnessScore, max: 100, label: "Compliance" }}
          icon={<TrendingUp className="h-4 w-4" />}
        />
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        {/* Left Column - Charts */}
        <div className="lg:col-span-2 space-y-6">
          {/* Quota Compliance Chart */}
          <QuotaChart 
            data={quotaData} 
            title="Diversity Quota Compliance"
            showLegend={true}
          />

          {/* Capacity Utilization */}
          <Card data-testid="card-capacity-chart">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BarChart className="h-5 w-5" />
                Capacity Utilization Over Time
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={capacityData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="month" />
                    <YAxis />
                    <Tooltip />
                    <Line 
                      type="monotone" 
                      dataKey="students" 
                      stroke="#3b82f6" 
                      strokeWidth={2}
                      name="Students"
                    />
                    <Line 
                      type="monotone" 
                      dataKey="internships" 
                      stroke="#10b981" 
                      strokeWidth={2}
                      name="Internships"
                    />
                    <Line 
                      type="monotone" 
                      dataKey="applications" 
                      stroke="#f59e0b" 
                      strokeWidth={2}
                      name="Applications"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

          {/* System Health */}
          <Card data-testid="card-system-health">
            <CardHeader>
              <CardTitle>System Health & Performance</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid gap-4 md:grid-cols-3">
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>AI Model Accuracy</span>
                    <span>94.2%</span>
                  </div>
                  <Progress value={94.2} className="h-2" />
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>System Uptime</span>
                    <span>99.9%</span>
                  </div>
                  <Progress value={99.9} className="h-2" />
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Data Quality Score</span>
                    <span>91.7%</span>
                  </div>
                  <Progress value={91.7} className="h-2" />
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Right Column - Alerts & Actions */}
        <div className="space-y-6">
          {/* Fairness Monitoring Alerts */}
          <Card data-testid="card-fairness-alerts">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-lg font-semibold">
                Fairness Alerts ({fairnessAlerts.length})
              </CardTitle>
              <AlertTriangle className="h-5 w-5 text-yellow-500" />
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {fairnessAlerts.map((alert) => (
                  <div
                    key={alert.id}
                    className="border border-border rounded-lg p-3 hover-elevate transition-all duration-200"
                    data-testid={`alert-${alert.id}`}
                  >
                    <div className="flex items-start justify-between mb-2">
                      <Badge className={getSeverityColor(alert.severity)}>
                        {alert.severity.toUpperCase()}
                      </Badge>
                      <span className="text-xs text-muted-foreground">
                        {new Date(alert.timestamp).toLocaleDateString()}
                      </span>
                    </div>
                    <p className="text-sm">{alert.message}</p>
                    <div className="flex items-center gap-2 mt-2">
                      <Button 
                        size="sm" 
                        variant="outline"
                        onClick={() => onViewDetails?.(alert.type)}
                        data-testid="button-view-alert"
                      >
                        View Details
                      </Button>
                    </div>
                  </div>
                ))}

                {fairnessAlerts.length === 0 && (
                  <div className="text-center py-4 text-muted-foreground">
                    <p>No active alerts</p>
                    <p className="text-sm">System is operating within fairness parameters</p>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Quick Actions */}
          <Card data-testid="card-quick-actions">
            <CardHeader>
              <CardTitle>Quick Actions</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <Button 
                className="w-full justify-start" 
                variant="outline"
                onClick={() => onViewDetails?.('students')}
                data-testid="button-view-students"
              >
                <Users className="h-4 w-4 mr-2" />
                View All Students
              </Button>
              <Button 
                className="w-full justify-start" 
                variant="outline"
                onClick={() => onViewDetails?.('employers')}
                data-testid="button-view-employers"
              >
                <Briefcase className="h-4 w-4 mr-2" />
                View All Employers
              </Button>
              <Button 
                className="w-full justify-start" 
                variant="outline"
                onClick={() => onViewDetails?.('matches')}
                data-testid="button-view-matches"
              >
                <Target className="h-4 w-4 mr-2" />
                AI Match Analysis
              </Button>
              <Button 
                className="w-full justify-start" 
                variant="outline"
                onClick={() => onExportData?.('compliance')}
                data-testid="button-export-compliance"
              >
                <Download className="h-4 w-4 mr-2" />
                Export Compliance Report
              </Button>
            </CardContent>
          </Card>

          {/* Recent Activity */}
          <Card data-testid="card-recent-activity">
            <CardHeader>
              <CardTitle>Recent Activity</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3 text-sm">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-green-500 rounded-full" />
                  <span>New employer registered: InnovateAI</span>
                  <span className="text-xs text-muted-foreground ml-auto">2m ago</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-blue-500 rounded-full" />
                  <span>25 new applications processed</span>
                  <span className="text-xs text-muted-foreground ml-auto">15m ago</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-yellow-500 rounded-full" />
                  <span>Quota compliance check completed</span>
                  <span className="text-xs text-muted-foreground ml-auto">1h ago</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-purple-500 rounded-full" />
                  <span>AI model retrained with new data</span>
                  <span className="text-xs text-muted-foreground ml-auto">3h ago</span>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}