import EmployerDashboard from '../EmployerDashboard';

export default function EmployerDashboardExample() {
  //todo: remove mock functionality
  const mockPostedInternships = [
    {
      id: "1",
      title: "Product Manager Intern",
      location: "San Francisco, CA",
      duration: "3 months",
      applications: 23,
      maxApplications: 30,
      status: "open" as const,
      deadline: "2024-02-15"
    },
    {
      id: "2", 
      title: "UX Research Intern",
      location: "San Francisco, CA",
      duration: "4 months",
      applications: 15,
      maxApplications: 20,
      status: "open" as const,
      deadline: "2024-02-20"
    },
    {
      id: "3",
      title: "Data Science Intern",
      location: "Remote",
      duration: "3 months", 
      applications: 45,
      maxApplications: 40,
      status: "filled" as const,
      deadline: "2024-01-30"
    }
  ];

  const mockCandidates = [
    {
      id: "1",
      name: "Sarah Chen",
      university: "Stanford University",
      major: "Computer Science & Business",
      location: "Palo Alto, CA",
      skills: ["Product Strategy", "Data Analysis", "Python", "Figma", "User Research"],
      matchScore: 92,
      matchReasons: [
        "Excellent academic background in CS + Business",
        "Previous PM internship experience",
        "Strong technical skills match",
        "Leadership experience in student organizations"
      ],
      status: "pending" as const,
      internshipId: "1"
    },
    {
      id: "2",
      name: "Alex Rodriguez", 
      university: "UC Berkeley",
      major: "Industrial Engineering",
      location: "Berkeley, CA",
      skills: ["Data Analysis", "Product Strategy", "SQL", "Tableau", "Agile"],
      matchScore: 87,
      matchReasons: [
        "Strong analytical background",
        "Experience with data visualization",
        "Good location match",
        "Relevant coursework in product management"
      ],
      status: "pending" as const,
      internshipId: "1"
    },
    {
      id: "3",
      name: "Maya Patel",
      university: "MIT",
      major: "Computer Science",
      location: "Cambridge, MA",
      skills: ["Machine Learning", "Python", "React", "Product Strategy", "Data Science"],
      matchScore: 89,
      matchReasons: [
        "Exceptional technical background",
        "Strong interest in product management",
        "Experience with ML products",
        "High academic achievement"
      ],
      status: "accepted" as const,
      internshipId: "2"
    },
    {
      id: "4",
      name: "Jordan Kim",
      university: "University of Washington",
      major: "Human-Computer Interaction",
      location: "Seattle, WA",
      skills: ["User Research", "Figma", "Product Strategy", "Statistics", "A/B Testing"],
      matchScore: 85,
      matchReasons: [
        "Perfect UX research background",
        "Strong product intuition",
        "Experience with user testing",
        "Good statistical analysis skills"
      ],
      status: "pending" as const,
      internshipId: "2"
    }
  ];

  return (
    <div className="min-h-screen bg-background">
      <EmployerDashboard
        companyName="TechFlow Solutions"
        postedInternships={mockPostedInternships}
        recommendedCandidates={mockCandidates}
        onCreateInternship={() => console.log('Create internship clicked')}
        onViewInternship={(id) => console.log('View internship:', id)}
        onAcceptCandidate={(id) => console.log('Accept candidate:', id)}
        onRejectCandidate={(id) => console.log('Reject candidate:', id)}
        onBulkAction={(ids, action) => console.log('Bulk action:', action, 'for candidates:', ids)}
      />
    </div>
  );
}