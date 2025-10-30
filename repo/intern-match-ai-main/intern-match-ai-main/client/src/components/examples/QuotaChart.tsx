import QuotaChart from '../QuotaChart';

export default function QuotaChartExample() {
  //todo: remove mock functionality
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

  return (
    <div className="p-6 bg-background">
      <QuotaChart data={mockQuotaData} title="Internship Quota Compliance" />
    </div>
  );
}