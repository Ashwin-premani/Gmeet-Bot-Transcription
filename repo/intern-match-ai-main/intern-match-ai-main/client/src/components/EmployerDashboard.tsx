import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Checkbox } from "@/components/ui/checkbox";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import MatchCard from "./MatchCard";
import MetricsCard from "./MetricsCard";
import { Plus, Search, Filter, Users, Briefcase, Eye, Download } from "lucide-react";

interface PostedInternship {
  id: string;
  title: string;
  location: string;
  duration: string;
  applications: number;
  maxApplications: number;
  status: 'open' | 'closed' | 'filled';
  deadline: string;
}

interface AIAnalysis {
  overallMatch: number;
  confidence: number;
  keyStrengths: string[];
  potentialConcerns: string[];
  skillGaps: string[];
  careerImpact: string;
  employerBenefits: string[];
  actionableAdvice: string[];
  breakdown: {
    skillsMatch: number;
    experienceMatch: number;
    locationMatch: number;
    cultureMatch: number;
    careerFitMatch: number;
  };
}

interface RecommendedCandidate {
  id: string;
  name: string;
  university: string;
  major: string;
  location: string;
  skills: string[];
  matchScore: number;
  matchReasons: string[];
  status: 'pending' | 'accepted' | 'rejected';
  internshipId: string;
  aiAnalysis?: AIAnalysis;
}

interface EmployerDashboardProps {
  companyName: string;
  postedInternships: PostedInternship[];
  recommendedCandidates: RecommendedCandidate[];
  onCreateInternship?: () => void;
  onViewInternship?: (internshipId: string) => void;
  onAcceptCandidate?: (candidateId: string) => void;
  onRejectCandidate?: (candidateId: string) => void;
  onBulkAction?: (candidateIds: string[], action: string) => void;
}

export default function EmployerDashboard({
  companyName,
  postedInternships,
  recommendedCandidates,
  onCreateInternship,
  onViewInternship,
  onAcceptCandidate,
  onRejectCandidate,
  onBulkAction
}: EmployerDashboardProps) {
  const [selectedCandidates, setSelectedCandidates] = useState<string[]>([]);
  const [searchTerm, setSearchTerm] = useState("");
  const [filterStatus, setFilterStatus] = useState("all");

  const totalApplications = postedInternships.reduce((sum, intern) => sum + intern.applications, 0);
  const activeInternships = postedInternships.filter(intern => intern.status === 'open').length;
  const pendingReviews = recommendedCandidates.filter(candidate => candidate.status === 'pending').length;
  
  const filteredCandidates = recommendedCandidates.filter(candidate => {
    const matchesSearch = candidate.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         candidate.university.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesStatus = filterStatus === 'all' || candidate.status === filterStatus;
    return matchesSearch && matchesStatus;
  });

  const handleSelectCandidate = (candidateId: string, checked: boolean) => {
    if (checked) {
      setSelectedCandidates([...selectedCandidates, candidateId]);
    } else {
      setSelectedCandidates(selectedCandidates.filter(id => id !== candidateId));
    }
  };

  const handleSelectAll = (checked: boolean) => {
    if (checked) {
      setSelectedCandidates(filteredCandidates.map(c => c.id));
    } else {
      setSelectedCandidates([]);
    }
  };

  return (
    <div className="space-y-6 p-6" data-testid="dashboard-employer">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">{companyName} Dashboard</h1>
          <p className="text-muted-foreground">Manage internships and review candidates</p>
        </div>
        <Button onClick={onCreateInternship} data-testid="button-create-internship">
          <Plus className="h-4 w-4 mr-2" />
          Post New Internship
        </Button>
      </div>

      {/* Metrics Row */}
      <div className="grid gap-4 md:grid-cols-4">
        <MetricsCard
          title="Total Applications"
          value={totalApplications}
          description="Across all positions"
          icon={<Users className="h-4 w-4" />}
        />
        <MetricsCard
          title="Active Internships"
          value={activeInternships}
          description="Currently hiring"
          icon={<Briefcase className="h-4 w-4" />}
        />
        <MetricsCard
          title="Pending Reviews"
          value={pendingReviews}
          description="Awaiting decision"
          icon={<Eye className="h-4 w-4" />}
        />
        <MetricsCard
          title="Match Success Rate"
          value="84%"
          description="Accepted recommendations"
          trend={{ value: 5, label: "from last month", direction: "up" }}
          icon={<Users className="h-4 w-4" />}
        />
      </div>

      <div className="grid gap-6 md:grid-cols-5">
        {/* Left Column - Posted Internships */}
        <div className="md:col-span-2 space-y-6">
          <Card data-testid="card-posted-internships">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Briefcase className="h-5 w-5" />
                Posted Internships ({postedInternships.length})
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {postedInternships.map((internship) => {
                  const progressPercentage = (internship.applications / internship.maxApplications) * 100;
                  
                  return (
                    <div
                      key={internship.id}
                      className="border border-border rounded-lg p-4 hover-elevate transition-all duration-200"
                      data-testid={`internship-${internship.id}`}
                    >
                      <div className="flex items-start justify-between mb-2">
                        <div className="flex-1">
                          <h4 className="font-medium">{internship.title}</h4>
                          <p className="text-sm text-muted-foreground">{internship.location}</p>
                        </div>
                        <Badge
                          className={
                            internship.status === 'open' 
                              ? "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200"
                              : internship.status === 'filled'
                              ? "bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200"
                              : "bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200"
                          }
                        >
                          {internship.status}
                        </Badge>
                      </div>

                      <div className="space-y-2">
                        <div className="flex justify-between text-sm">
                          <span>Applications</span>
                          <span>{internship.applications}/{internship.maxApplications}</span>
                        </div>
                        <div className="w-full bg-muted rounded-full h-2">
                          <div 
                            className="bg-primary h-2 rounded-full transition-all duration-300"
                            style={{ width: `${progressPercentage}%` }}
                          />
                        </div>
                      </div>

                      <div className="flex justify-between items-center mt-3">
                        <span className="text-xs text-muted-foreground">
                          Deadline: {new Date(internship.deadline).toLocaleDateString()}
                        </span>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => onViewInternship?.(internship.id)}
                          data-testid="button-view-internship"
                        >
                          <Eye className="h-4 w-4 mr-1" />
                          View
                        </Button>
                      </div>
                    </div>
                  );
                })}

                {postedInternships.length === 0 && (
                  <div className="text-center py-8 text-muted-foreground">
                    <p>No internships posted yet</p>
                    <p className="text-sm">Create your first internship to start receiving applications</p>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Right Column - Recommended Candidates */}
        <div className="md:col-span-3 space-y-6">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-semibold">AI-Recommended Candidates</h2>
            <div className="flex items-center gap-2">
              <Input
                placeholder="Search candidates..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-48"
                data-testid="input-search-candidates"
              />
              <Select value={filterStatus} onValueChange={setFilterStatus}>
                <SelectTrigger className="w-32" data-testid="select-filter-status">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Status</SelectItem>
                  <SelectItem value="pending">Pending</SelectItem>
                  <SelectItem value="accepted">Accepted</SelectItem>
                  <SelectItem value="rejected">Rejected</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          {/* Bulk Actions */}
          {selectedCandidates.length > 0 && (
            <Card className="bg-blue-50 dark:bg-blue-950 border-blue-200 dark:border-blue-800">
              <CardContent className="py-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-medium">
                      {selectedCandidates.length} candidate(s) selected
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Button 
                      size="sm" 
                      onClick={() => onBulkAction?.(selectedCandidates, 'accept')}
                      data-testid="button-bulk-accept"
                    >
                      Accept All
                    </Button>
                    <Button 
                      size="sm" 
                      variant="outline"
                      onClick={() => onBulkAction?.(selectedCandidates, 'reject')}
                      data-testid="button-bulk-reject"
                    >
                      Reject All
                    </Button>
                    <Button 
                      size="sm" 
                      variant="outline"
                      onClick={() => onBulkAction?.(selectedCandidates, 'export')}
                      data-testid="button-bulk-export"
                    >
                      <Download className="h-4 w-4 mr-1" />
                      Export
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Select All */}
          <div className="flex items-center gap-2">
            <Checkbox
              id="select-all"
              checked={filteredCandidates.length > 0 && selectedCandidates.length === filteredCandidates.length}
              onCheckedChange={handleSelectAll}
              data-testid="checkbox-select-all"
            />
            <label htmlFor="select-all" className="text-sm font-medium">
              Select all visible candidates
            </label>
          </div>

          {/* Candidates Grid */}
          <div className="space-y-4">
            {filteredCandidates.map((candidate) => (
              // console.log({candidate}),
              <div key={candidate.id} className="flex items-start gap-3">
                <Checkbox
                  checked={selectedCandidates.includes(candidate.id)}
                  onCheckedChange={(checked) => handleSelectCandidate(candidate.id, checked as boolean)}
                  className="mt-4"
                  data-testid={`checkbox-candidate-${candidate.id}`}
                />
                <div className="flex-1">
                  <MatchCard
                    id={candidate.id}
                    type="candidate"
                    candidateName={candidate.name}
                    university={candidate.university}
                    title={candidate.major}
                    location={candidate.location}
                    matchScore={candidate.matchScore}
                    matchReasons={candidate.matchReasons}
                    skills={candidate.skills}
                    status={candidate.status}
                    onAccept={() => onAcceptCandidate?.(candidate.id)}
                    onReject={() => onRejectCandidate?.(candidate.id)}
                    aiAnalysis={candidate.aiAnalysis}
                  />
                </div>
              </div>
            ))}
          </div>

          {filteredCandidates.length === 0 && (
            <Card>
              <CardContent className="py-8 text-center">
                <p className="text-muted-foreground">No candidates found</p>
                <p className="text-sm text-muted-foreground mt-1">
                  {searchTerm ? "Try adjusting your search criteria" : "Post internships to receive candidate recommendations"}
                </p>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}