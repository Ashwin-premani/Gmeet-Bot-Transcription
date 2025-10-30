import { useParams, useLocation } from "wouter";
import { useQuery } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Progress } from "@/components/ui/progress";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { ArrowLeft, MapPin, Calendar, DollarSign, Building, Clock, Users, Target, CheckCircle, FileText, GraduationCap, Award, ExternalLink } from "lucide-react";
import { getEmployerInternshipDetails, me } from "@/lib/api";

export default function EmployerInternshipDetailsPage() {
  const { id } = useParams();
  const [, setLocation] = useLocation();
  
  const { data: auth } = useQuery({ queryKey: ["auth", "me"], queryFn: me });
  const { data: internship, isLoading } = useQuery({
    queryKey: ["employer", "internship", id],
    queryFn: () => getEmployerInternshipDetails(id!),
    enabled: !!id,
  });

  const handleBack = () => {
    setLocation("/employer");
  };

  const handleEditInternship = () => {
    // TODO: Implement edit functionality
    console.log("Edit internship:", id);
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-background p-6">
        <div className="max-w-6xl mx-auto">
          <div className="text-lg">Loading internship details...</div>
        </div>
      </div>
    );
  }

  if (!internship) {
    return (
      <div className="min-h-screen bg-background p-6">
        <div className="max-w-6xl mx-auto">
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
  const pendingApplications = internship.applications?.filter(app => app.status === 'pending').length || 0;
  const acceptedApplications = internship.applications?.filter(app => app.status === 'accepted').length || 0;
  const rejectedApplications = internship.applications?.filter(app => app.status === 'rejected').length || 0;

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'pending': return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200';
      case 'accepted': return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200';
      case 'rejected': return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200';
      default: return 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200';
    }
  };

  return (
    <div className="min-h-screen bg-background p-6">
      <div className="max-w-6xl mx-auto">
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
                <span>{internship.employer.companyName}</span>
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
                {internship.stipend && (
                  <div className="flex items-center gap-1">
                    <DollarSign className="h-4 w-4" />
                    <span>{internship.stipend}</span>
                  </div>
                )}
                <div className="flex items-center gap-1">
                  <Clock className="h-4 w-4" />
                  <span>Posted {new Date(internship.createdAt).toLocaleDateString()}</span>
                </div>
              </div>
            </div>
            
            <div className="flex gap-2">
              <Button onClick={handleEditInternship} variant="outline">
                <FileText className="h-4 w-4 mr-2" />
                Edit Internship
              </Button>
              <Badge className={internship.status === 'open' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}>
                {internship.status}
              </Badge>
            </div>
          </div>
          
          {/* Application Statistics */}
          <div className="grid gap-4 md:grid-cols-4 mb-6">
            <Card>
              <CardContent className="p-4 text-center">
                <div className="text-2xl font-bold text-blue-600">{internship.applications?.length || 0}</div>
                <div className="text-sm text-muted-foreground">Total Applications</div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-4 text-center">
                <div className="text-2xl font-bold text-yellow-600">{pendingApplications}</div>
                <div className="text-sm text-muted-foreground">Pending Review</div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-4 text-center">
                <div className="text-2xl font-bold text-green-600">{acceptedApplications}</div>
                <div className="text-sm text-muted-foreground">Accepted</div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-4 text-center">
                <div className="text-2xl font-bold text-red-600">{rejectedApplications}</div>
                <div className="text-sm text-muted-foreground">Rejected</div>
              </CardContent>
            </Card>
          </div>

          {/* Application Progress */}
          <Card className="mb-6">
            <CardContent className="p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="font-medium">Application Progress</span>
                <div className="flex items-center gap-2">
                  <Users className="h-4 w-4" />
                  <span className="text-sm">{internship.currentApplications}/{internship.maxApplications} slots filled</span>
                </div>
              </div>
              <Progress value={applicationProgress} className="mb-2" />
              <div className="flex justify-between text-sm text-muted-foreground">
                <span>Deadline: {new Date(internship.deadline).toLocaleDateString()}</span>
                <span>{Math.round(applicationProgress)}% capacity reached</span>
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="grid gap-6 lg:grid-cols-3">
          {/* Left Column - Internship Details */}
          <div className="lg:col-span-2 space-y-6">
            {/* Description */}
            <Card>
              <CardHeader>
                <CardTitle>Job Description</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm leading-relaxed whitespace-pre-wrap">{internship.description}</p>
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

            {/* Applications Table */}
            <Card>
              <CardHeader>
                <CardTitle>Applications ({internship.applications?.length || 0})</CardTitle>
              </CardHeader>
              <CardContent>
                {internship.applications && internship.applications.length > 0 ? (
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Student</TableHead>
                        <TableHead>University</TableHead>
                        <TableHead>Major</TableHead>
                        <TableHead>Match Score</TableHead>
                        <TableHead>Status</TableHead>
                        <TableHead>Applied</TableHead>
                        <TableHead>Actions</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {internship.applications.map((application) => (
                        <TableRow key={application.id}>
                          <TableCell className="font-medium">
                            <div className="flex items-center gap-2">
                              <div>
                                <div className="font-medium">Student #{application.student.id.slice(-4)}</div>
                                <div className="text-xs text-muted-foreground">{application.student.location}</div>
                              </div>
                            </div>
                          </TableCell>
                          <TableCell>{application.student.university}</TableCell>
                          <TableCell>{application.student.major}</TableCell>
                          <TableCell>
                            {application.aiMatchScore && (
                              <div className="flex items-center gap-2">
                                <div className="text-sm font-medium">{application.aiMatchScore}%</div>
                                <Progress value={application.aiMatchScore} className="w-16 h-2" />
                              </div>
                            )}
                          </TableCell>
                          <TableCell>
                            <Badge className={getStatusColor(application.status)}>
                              {application.status}
                            </Badge>
                          </TableCell>
                          <TableCell className="text-xs">
                            {new Date(application.appliedAt).toLocaleDateString()}
                          </TableCell>
                          <TableCell>
                            <div className="flex gap-1">
                              <Button size="sm" variant="outline">
                                <ExternalLink className="h-3 w-3 mr-1" />
                                View
                              </Button>
                              {application.status === 'pending' && (
                                <>
                                  <Button size="sm" variant="outline" className="text-green-600">
                                    Accept
                                  </Button>
                                  <Button size="sm" variant="outline" className="text-red-600">
                                    Reject
                                  </Button>
                                </>
                              )}
                            </div>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                ) : (
                  <div className="text-center py-8 text-muted-foreground">
                    <Users className="h-12 w-12 mx-auto mb-4 opacity-50" />
                    <p>No applications received yet</p>
                    <p className="text-sm">Applications will appear here when students apply</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Right Column - Sidebar */}
          <div className="space-y-6">
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

            {/* Company Information */}
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">About {internship.employer.companyName}</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex items-center gap-2 text-sm">
                    <Building className="h-4 w-4 text-muted-foreground" />
                    <span>{internship.employer.industry}</span>
                  </div>
                  <div className="flex items-center gap-2 text-sm">
                    <MapPin className="h-4 w-4 text-muted-foreground" />
                    <span>{internship.employer.location}</span>
                  </div>
                  {internship.employer.description && (
                    <p className="text-sm text-muted-foreground leading-relaxed">
                      {internship.employer.description}
                    </p>
                  )}
                </div>
              </CardContent>
            </Card>

            {/* Quick Actions */}
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Quick Actions</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <Button className="w-full" variant="outline">
                  <FileText className="h-4 w-4 mr-2" />
                  Export Applications
                </Button>
                <Button className="w-full" variant="outline">
                  <Users className="h-4 w-4 mr-2" />
                  View All Candidates
                </Button>
                <Button className="w-full" variant="outline">
                  <Target className="h-4 w-4 mr-2" />
                  Extend Deadline
                </Button>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}
