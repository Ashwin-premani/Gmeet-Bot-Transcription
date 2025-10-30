import AdminDashboard from '../AdminDashboard';

export default function AdminDashboardExample() {
  //todo: remove mock functionality
  const mockMetrics = {
    totalStudents: 2456,
    totalEmployers: 156,
    totalInternships: 89,
    totalApplications: 1234,
    successfulMatches: 847,
    averageMatchScore: 84.2,
    quotaCompliance: 87,
    fairnessScore: 92
  };

  const mockQuotaData = [
    {
      category: "Engineering",
      allocated: 120,
      total: 150,
      percentage: 80,
      color: "#3b82f6"
    },
    {
      category: "Product Management",
      allocated: 45,
      total: 60,
      percentage: 75,
      color: "#10b981"
    },
    {
      category: "Design",
      allocated: 25,
      total: 40,
      percentage: 62.5,
      color: "#f59e0b"
    },
    {
      category: "Marketing", 
      allocated: 30,
      total: 35,
      percentage: 85.7,
      color: "#ef4444"
    },
    {
      category: "Data Science",
      allocated: 18,
      total: 25,
      percentage: 72,
      color: "#8b5cf6"
    }
  ];

  const mockCapacityData = [
    { month: "Sep", students: 1850, internships: 45, applications: 780 },
    { month: "Oct", students: 2100, internships: 52, applications: 890 },
    { month: "Nov", students: 2250, internships: 67, applications: 1050 },
    { month: "Dec", students: 2350, internships: 73, applications: 1150 },
    { month: "Jan", students: 2456, internships: 89, applications: 1234 },
    { month: "Feb", students: 2580, internships: 95, applications: 1340 }
  ];

  const mockFairnessAlerts = [
    {
      id: "1",
      type: "quota" as const,
      severity: "medium" as const,
      message: "Engineering quota at 80% - consider increasing diversity outreach",
      timestamp: "2024-01-25T10:30:00Z"
    },
    {
      id: "2", 
      type: "bias" as const,
      severity: "low" as const,
      message: "Minor bias detected in university preference algorithm",
      timestamp: "2024-01-24T15:45:00Z"
    },
    {
      id: "3",
      type: "capacity" as const,
      severity: "high" as const,
      message: "Product Management internships oversubscribed by 150%",
      timestamp: "2024-01-23T09:15:00Z"
    }
  ];

  return (
    <div className="min-h-screen bg-background">
      <AdminDashboard
        metrics={mockMetrics}
        quotaData={mockQuotaData}
        capacityData={mockCapacityData}
        fairnessAlerts={mockFairnessAlerts}
        onExportData={(type) => console.log('Export data:', type)}
        onViewDetails={(type) => console.log('View details:', type)}
      />
    </div>
  );
}