import { useParams, useLocation } from "wouter";
import { useQuery } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Progress } from "@/components/ui/progress";
import { ArrowLeft, MapPin, Calendar, DollarSign, Building, Clock, Users, Target, CheckCircle } from "lucide-react";
import { me } from "@/lib/api";

// Mock internship details - replace with real API call
const getInternshipDetails = async (id: string) => {
  // This would be a real API call
  return {
    id,
    title: "Product Manager Intern",
    company: "TechFlow Solutions",
    location: "San Francisco, CA",
    duration: "3 months",
    stipend: "$3,000/month",
    startDate: "2024-06-01",
    applicationDeadline: "2024-02-15",
    status: "open",
    maxApplications: 30,
    currentApplications: 23,
    matchScore: 87,
    matchReasons: [
      "Strong product management coursework match",
      "Previous experience with user research",
      "Skills align with job requirements",
      "Location preference matches"
    ],
    description: "Join our dynamic product team and help shape the future of our SaaS platform. You'll work directly with senior product managers to analyze user data, conduct research, and contribute to product roadmap decisions. This role offers hands-on experience with product strategy, user experience design, and agile development processes.",
    responsibilities: [
      "Assist in conducting user research and analyzing feedback",
      "Support product roadmap planning and feature prioritization",
      "Collaborate with engineering and design teams on feature development",
      "Create and maintain product documentation",
      "Participate in user testing sessions and analyze results",
      "Help monitor product metrics and KPIs"
    ],
    requirements: [
      "Currently pursuing a degree in Business, Computer Science, or related field",
      "Strong analytical and problem-solving skills",
      "Excellent written and verbal communication",
      "Experience with data analysis tools (Excel, SQL preferred)",
      "Interest in user experience and product design",
      "Previous internship or project experience preferred"
    ],
    skills: ["Product Strategy", "Data Analysis", "User Research", "SQL", "Figma", "Agile"],
    aboutCompany: "TechFlow Solutions is a leading SaaS company providing workflow automation tools for enterprise customers. Founded in 2018, we've grown to serve over 10,000 companies worldwide with our innovative platform.",
    benefits: [
      "Competitive stipend and performance bonuses",
      "Mentorship from senior product leaders",
      "Access to premium learning resources and conferences",
      "Flexible work arrangements",
      "Full-time job consideration upon graduation"
    ]
  };
};

export default function InternshipDetailsPage() {
  const { id } = useParams();
  const [, setLocation] = useLocation();
  
  const { data: auth } = useQuery({ queryKey: ["auth", "me"], queryFn: me });
  const { data: internship, isLoading } = useQuery({
    queryKey: ["internship", id],
    queryFn: () => getInternshipDetails(id!),
    enabled: !!id,
  });

  const handleApply = () => {
    // TODO: Implement application logic
    console.log("Apply to internship:", id);
  };

  const handleBack = () => {
    setLocation("/student");
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-background p-6">
        <div className="max-w-4xl mx-auto">
          <div className="text-lg">Loading internship details...</div>
        </div>
      </div>
    );
  }

  if (!internship) {
    return (
      <div className="min-h-screen bg-background p-6">
        <div className="max-w-4xl mx-auto">
          <Button variant="ghost" onClick={handleBack} className="mb-4">
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back to Dashboard
          </Button>
          <div className="text-lg">Internship not found</div>
        </div>
      </div>
    );
  }

  const applicationProgress = (internship.currentApplications / internship.maxApplications) * 100;

  return (
    <div className="min-h-screen bg-background p-6">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="mb-6">
          <Button variant="ghost" onClick={handleBack} className="mb-4">
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back to Dashboard
          </Button>
          
          <div className="flex items-start justify-between mb-4">
            <div className="flex-1">
              <h1 className="text-3xl font-bold mb-2">{internship.title}</h1>
              <div className="flex items-center gap-2 text-lg text-muted-foreground mb-4">
                <Building className="h-5 w-5" />
                <span>{internship.company}</span>
              </div>
              
              <div className="flex flex-wrap gap-4 text-sm text-muted-foreground">
                <div className="flex items-center gap-1">
                  <MapPin className="h-4 w-4" />
                  <span>{internship.location}</span>
                </div>
                <div className="flex items-center gap-1">
                  <Calendar className="h-4 w-4" />
                  <span>{internship.duration}</span>
                </div>
                <div className="flex items-center gap-1">
                  <DollarSign className="h-4 w-4" />
                  <span>{internship.stipend}</span>
                </div>
                <div className="flex items-center gap-1">
                  <Clock className="h-4 w-4" />
                  <span>Starts {new Date(internship.startDate).toLocaleDateString()}</span>
                </div>
              </div>
            </div>
            
            {/* AI Match Score */}
            <Card className="w-48">
              <CardContent className="p-4 text-center">
                <div className="text-3xl font-bold text-green-600 mb-2">{internship.matchScore}%</div>
                <div className="text-sm text-muted-foreground mb-3">AI Match Score</div>
                <Progress value={internship.matchScore} className="mb-3" />
                <div className="text-xs text-left">
                  <div className="font-medium mb-1">Match Reasons:</div>
                  <ul className="space-y-1">
                    {internship.matchReasons.map((reason, index) => (
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
          
          {/* Application Status */}
          <Card className="mb-6">
            <CardContent className="p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="font-medium">Application Status</span>
                <div className="flex items-center gap-2">
                  <Users className="h-4 w-4" />
                  <span className="text-sm">{internship.currentApplications}/{internship.maxApplications} applications</span>
                </div>
              </div>
              <Progress value={applicationProgress} className="mb-2" />
              <div className="flex justify-between text-sm text-muted-foreground">
                <span>Deadline: {new Date(internship.applicationDeadline).toLocaleDateString()}</span>
                <Badge className={internship.status === 'open' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}>
                  {internship.status}
                </Badge>
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="grid gap-6 md:grid-cols-3">
          {/* Left Column - Main Content */}
          <div className="md:col-span-2 space-y-6">
            {/* Description */}
            <Card>
              <CardHeader>
                <CardTitle>Role Description</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm leading-relaxed">{internship.description}</p>
              </CardContent>
            </Card>

            {/* Responsibilities */}
            <Card>
              <CardHeader>
                <CardTitle>Key Responsibilities</CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="space-y-2">
                  {internship.responsibilities.map((responsibility, index) => (
                    <li key={index} className="flex items-start gap-2 text-sm">
                      <Target className="h-4 w-4 text-blue-500 mt-0.5 flex-shrink-0" />
                      <span>{responsibility}</span>
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>

            {/* Requirements */}
            <Card>
              <CardHeader>
                <CardTitle>Requirements</CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="space-y-2">
                  {internship.requirements.map((requirement, index) => (
                    <li key={index} className="flex items-start gap-2 text-sm">
                      <CheckCircle className="h-4 w-4 text-green-500 mt-0.5 flex-shrink-0" />
                      <span>{requirement}</span>
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>
          </div>

          {/* Right Column - Sidebar */}
          <div className="space-y-6">
            {/* Apply Button */}
            <Card>
              <CardContent className="p-4">
                <Button 
                  onClick={handleApply} 
                  className="w-full mb-4"
                  disabled={internship.status !== 'open'}
                >
                  {internship.status === 'open' ? 'Apply Now' : 'Applications Closed'}
                </Button>
                <p className="text-xs text-muted-foreground text-center">
                  Application deadline: {new Date(internship.applicationDeadline).toLocaleDateString()}
                </p>
              </CardContent>
            </Card>

            {/* Required Skills */}
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Required Skills</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex flex-wrap gap-2">
                  {internship.skills.map((skill, index) => (
                    <Badge key={index} variant="secondary" className="text-xs">
                      {skill}
                    </Badge>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* About Company */}
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">About {internship.company}</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground leading-relaxed mb-4">
                  {internship.aboutCompany}
                </p>
              </CardContent>
            </Card>

            {/* Benefits */}
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">What We Offer</CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="space-y-2">
                  {internship.benefits.map((benefit, index) => (
                    <li key={index} className="flex items-start gap-2 text-sm">
                      <CheckCircle className="h-4 w-4 text-green-500 mt-0.5 flex-shrink-0" />
                      <span>{benefit}</span>
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}
