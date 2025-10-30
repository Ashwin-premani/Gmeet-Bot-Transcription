import ApplicationTracker from '../ApplicationTracker';

export default function ApplicationTrackerExample() {
  //todo: remove mock functionality
  const mockApplications = [
    {
      id: "1",
      internshipTitle: "Product Manager Intern",
      company: "TechFlow Solutions",
      location: "San Francisco, CA",
      appliedDate: "2024-01-15",
      status: "pending" as const,
      matchScore: 87,
      deadline: "2024-02-15",
    },
    {
      id: "2", 
      internshipTitle: "UX Research Intern",
      company: "Design Labs",
      location: "New York, NY",
      appliedDate: "2024-01-10",
      status: "accepted" as const,
      matchScore: 92,
      response: "Congratulations! We're excited to offer you the position. Please respond by Feb 1st."
    },
    {
      id: "3",
      internshipTitle: "Data Science Intern",
      company: "Analytics Pro",
      location: "Seattle, WA", 
      appliedDate: "2024-01-08",
      status: "rejected" as const,
      matchScore: 74,
      response: "Thank you for your interest. Unfortunately, we've decided to move forward with other candidates."
    },
    {
      id: "4",
      internshipTitle: "Software Engineering Intern",
      company: "CloudTech",
      location: "Austin, TX",
      appliedDate: "2024-01-20",
      status: "pending" as const,
      matchScore: 81,
      deadline: "2024-01-25",
    }
  ];

  return (
    <div className="p-6 bg-background">
      <ApplicationTracker
        applications={mockApplications}
        onViewDetails={(id) => console.log('View details for:', id)}
        onWithdraw={(id) => console.log('Withdraw application:', id)}
      />
    </div>
  );
}