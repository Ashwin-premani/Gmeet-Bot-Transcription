import StudentDashboard from '../StudentDashboard';

export default function StudentDashboardExample() {
  //todo: remove mock functionality
  const mockProfile = {
    id: "1",
    name: "Sarah Chen",
    email: "sarah.chen@stanford.edu",
    university: "Stanford University",
    major: "Computer Science & Business",
    graduationYear: 2025,
    gpa: "3.8",
    skills: ["Product Strategy", "Data Analysis", "Python", "Figma", "User Research", "Agile", "SQL", "React"],
    interests: ["Product Management", "AI/ML", "User Experience", "Startups"],
    location: "Palo Alto, CA"
  };

  const mockInternships = [
    {
      id: "1",
      title: "Product Manager Intern",
      company: "TechFlow Solutions",
      location: "San Francisco, CA",
      duration: "3 months",
      stipend: "$3,000/month",
      skills: ["Product Strategy", "User Research", "Data Analysis", "Agile"],
      matchScore: 87,
      matchReasons: [
        "Strong product management coursework match",
        "Previous experience with user research",
        "Skills align with job requirements",
        "Location preference matches"
      ],
      status: "open" as const
    },
    {
      id: "2",
      title: "AI Product Intern",
      company: "InnovateAI",
      location: "Palo Alto, CA",
      duration: "4 months",
      stipend: "$3,500/month",
      skills: ["Machine Learning", "Product Strategy", "Python", "Data Analysis"],
      matchScore: 92,
      matchReasons: [
        "Excellent AI/ML background match",
        "Strong technical and business combination",
        "Perfect location preference",
        "High academic performance alignment"
      ],
      status: "open" as const
    },
    {
      id: "3",
      title: "UX Research Intern",
      company: "Design Labs",
      location: "San Francisco, CA", 
      duration: "3 months",
      stipend: "$2,800/month",
      skills: ["User Research", "Figma", "Data Analysis", "Product Strategy"],
      matchScore: 78,
      matchReasons: [
        "Strong UX research interest",
        "Figma skills match perfectly",
        "Product strategy background",
        "Location works well"
      ],
      status: "open" as const
    }
  ];

  const mockApplications = [
    {
      id: "1",
      internshipTitle: "Product Manager Intern",
      company: "TechFlow Solutions",
      location: "San Francisco, CA",
      appliedDate: "2024-01-15",
      status: "pending",
      matchScore: 87,
      deadline: "2024-02-15"
    },
    {
      id: "2",
      internshipTitle: "UX Research Intern", 
      company: "Design Labs",
      location: "New York, NY",
      appliedDate: "2024-01-10",
      status: "accepted",
      matchScore: 92,
      response: "Congratulations! We're excited to offer you the position."
    }
  ];

  return (
    <div className="min-h-screen bg-background">
      <StudentDashboard
        profile={mockProfile}
        recommendedInternships={mockInternships}
        applications={mockApplications}
        onApplyToInternship={(id) => console.log('Apply to internship:', id)}
        onUpdateProfile={() => console.log('Update profile clicked')}
      />
    </div>
  );
}