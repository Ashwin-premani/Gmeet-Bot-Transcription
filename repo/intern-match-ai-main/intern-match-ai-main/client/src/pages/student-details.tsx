import { useParams, useLocation } from "wouter";
import { useQuery } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Progress } from "@/components/ui/progress";
import { ArrowLeft, GraduationCap, MapPin, Star, Mail, Calendar, Award, CheckCircle, Download } from "lucide-react";
import { me } from "@/lib/api";

// Mock student details - replace with real API call
const getStudentDetails = async (id: string) => {
  // This would be a real API call
  return {
    id,
    name: "Sarah Chen",
    email: "sarah.chen@stanford.edu",
    university: "Stanford University",
    major: "Computer Science & Business",
    graduationYear: 2025,
    location: "Palo Alto, CA",
    gpa: "3.8",
    matchScore: 92,
    matchReasons: [
      "Excellent academic background in CS + Business",
      "Previous PM internship experience",
      "Strong technical skills match",
      "Leadership experience in student organizations"
    ],
    skills: ["Product Strategy", "Data Analysis", "Python", "Figma", "User Research", "Agile", "SQL", "React"],
    interests: ["Product Management", "AI/ML", "User Experience", "Startups"],
    experience: [
      {
        title: "Product Intern",
        company: "StartupXYZ",
        duration: "Summer 2023",
        description: "Led user research for mobile app redesign, resulting in 25% increase in user engagement"
      },
      {
        title: "Research Assistant",
        company: "Stanford HCI Lab",
        duration: "2022-2023",
        description: "Conducted user studies on AI-human interaction, published 2 research papers"
      }
    ],
    projects: [
      {
        title: "EcoTrack - Sustainability App",
        description: "Built full-stack web app for tracking personal carbon footprint with React and Node.js",
        technologies: ["React", "Node.js", "MongoDB", "Firebase"]
      },
      {
        title: "ML Recommendation Engine",
        description: "Developed collaborative filtering algorithm for course recommendations with 85% accuracy",
        technologies: ["Python", "Scikit-learn", "Pandas", "Flask"]
      }
    ],
    achievements: [
      "Dean's List - Fall 2022, Spring 2023",
      "President, Stanford Product Management Club",
      "Winner, Stanford TreeHacks Hackathon 2023",
      "Google CSSI Scholar"
    ],
    languages: ["English (Native)", "Mandarin (Fluent)", "Spanish (Conversational)"],
    availability: "Available for summer 2024 internships",
    resumeUrl: "/resumes/sarah-chen.pdf"
  };
};

export default function StudentDetailsPage() {
  const { id } = useParams();
  const [, setLocation] = useLocation();
  
  const { data: auth } = useQuery({ queryKey: ["auth", "me"], queryFn: me });
  const { data: student, isLoading } = useQuery({
    queryKey: ["student", id],
    queryFn: () => getStudentDetails(id!),
    enabled: !!id,
  });

  const handleAccept = () => {
    // TODO: Implement accept logic
    console.log("Accept student:", id);
  };

  const handleReject = () => {
    // TODO: Implement reject logic
    console.log("Reject student:", id);
  };

  const handleBack = () => {
    setLocation("/employer");
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-background p-6">
        <div className="max-w-4xl mx-auto">
          <div className="text-lg">Loading student details...</div>
        </div>
      </div>
    );
  }

  if (!student) {
    return (
      <div className="min-h-screen bg-background p-6">
        <div className="max-w-4xl mx-auto">
          <Button variant="ghost" onClick={handleBack} className="mb-4">
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back to Dashboard
          </Button>
          <div className="text-lg">Student not found</div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background p-6">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="mb-6">
          <Button variant="ghost" onClick={handleBack} className="mb-4">
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back to Dashboard
          </Button>
          
          <div className="flex items-start gap-6 mb-6">
            {/* Profile Picture and Basic Info */}
            <div className="flex items-start gap-4">
              <Avatar className="h-24 w-24">
                <AvatarImage src="" />
                <AvatarFallback className="text-lg font-semibold">
                  {student.name.split(' ').map(n => n[0]).join('')}
                </AvatarFallback>
              </Avatar>
              
              <div className="flex-1">
                <h1 className="text-3xl font-bold mb-2">{student.name}</h1>
                <div className="space-y-1 text-muted-foreground">
                  <div className="flex items-center gap-2">
                    <GraduationCap className="h-4 w-4" />
                    <span>{student.major} • {student.university}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Calendar className="h-4 w-4" />
                    <span>Graduating {student.graduationYear}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <MapPin className="h-4 w-4" />
                    <span>{student.location}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Mail className="h-4 w-4" />
                    <span>{student.email}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Star className="h-4 w-4" />
                    <span>GPA: {student.gpa}</span>
                  </div>
                </div>
              </div>
            </div>
            
            {/* AI Match Score */}
            <Card className="w-56">
              <CardContent className="p-4 text-center">
                <div className="text-3xl font-bold text-green-600 mb-2">{student.matchScore}%</div>
                <div className="text-sm text-muted-foreground mb-3">AI Match Score</div>
                <Progress value={student.matchScore} className="mb-3" />
                <div className="text-xs text-left">
                  <div className="font-medium mb-1">Match Reasons:</div>
                  <ul className="space-y-1">
                    {student.matchReasons.map((reason, index) => (
                      <li key={index} className="flex items-start gap-1">
                        <CheckCircle className="h-3 w-3 text-green-500 mt-0.5 flex-shrink-0" />
                        <span>{reason}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Action Buttons */}
          <div className="flex gap-3 mb-6">
            <Button onClick={handleAccept} className="flex-1">
              Accept Candidate
            </Button>
            <Button onClick={handleReject} variant="outline" className="flex-1">
              Reject Candidate
            </Button>
            <Button variant="outline" className="flex items-center gap-2">
              <Download className="h-4 w-4" />
              Download Resume
            </Button>
          </div>
        </div>

        <div className="grid gap-6 md:grid-cols-3">
          {/* Left Column - Main Content */}
          <div className="md:col-span-2 space-y-6">
            {/* Skills & Interests */}
            <Card>
              <CardHeader>
                <CardTitle>Skills & Interests</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <h4 className="font-medium mb-2">Technical Skills</h4>
                  <div className="flex flex-wrap gap-2">
                    {student.skills.map((skill, index) => (
                      <Badge key={index} variant="secondary">
                        {skill}
                      </Badge>
                    ))}
                  </div>
                </div>
                <div>
                  <h4 className="font-medium mb-2">Areas of Interest</h4>
                  <div className="flex flex-wrap gap-2">
                    {student.interests.map((interest, index) => (
                      <Badge key={index} variant="outline">
                        {interest}
                      </Badge>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Experience */}
            <Card>
              <CardHeader>
                <CardTitle>Experience</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {student.experience.map((exp, index) => (
                    <div key={index} className="border-l-2 border-muted pl-4">
                      <div className="flex items-center gap-2 mb-1">
                        <h4 className="font-medium">{exp.title}</h4>
                        <span className="text-sm text-muted-foreground">@ {exp.company}</span>
                      </div>
                      <div className="text-sm text-muted-foreground mb-2">{exp.duration}</div>
                      <p className="text-sm leading-relaxed">{exp.description}</p>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Projects */}
            <Card>
              <CardHeader>
                <CardTitle>Notable Projects</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {student.projects.map((project, index) => (
                    <div key={index} className="border-l-2 border-muted pl-4">
                      <h4 className="font-medium mb-2">{project.title}</h4>
                      <p className="text-sm leading-relaxed mb-2">{project.description}</p>
                      <div className="flex flex-wrap gap-1">
                        {project.technologies.map((tech, techIndex) => (
                          <Badge key={techIndex} variant="outline" className="text-xs">
                            {tech}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Right Column - Sidebar */}
          <div className="space-y-6">
            {/* Quick Actions */}
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Quick Actions</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <Button variant="outline" className="w-full justify-start">
                  <Mail className="h-4 w-4 mr-2" />
                  Send Message
                </Button>
                <Button variant="outline" className="w-full justify-start">
                  <Calendar className="h-4 w-4 mr-2" />
                  Schedule Interview
                </Button>
                <Button variant="outline" className="w-full justify-start">
                  <Download className="h-4 w-4 mr-2" />
                  Export Profile
                </Button>
              </CardContent>
            </Card>

            {/* Achievements */}
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Achievements</CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="space-y-2">
                  {student.achievements.map((achievement, index) => (
                    <li key={index} className="flex items-start gap-2 text-sm">
                      <Award className="h-4 w-4 text-yellow-500 mt-0.5 flex-shrink-0" />
                      <span>{achievement}</span>
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>

            {/* Languages */}
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Languages</CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="space-y-1">
                  {student.languages.map((language, index) => (
                    <li key={index} className="text-sm">{language}</li>
                  ))}
                </ul>
              </CardContent>
            </Card>

            {/* Availability */}
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Availability</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground">{student.availability}</p>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}
