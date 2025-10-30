import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import MatchCard from "./MatchCard";
import ApplicationTracker from "./ApplicationTracker";
import MetricsCard from "./MetricsCard";
import { Search, User, GraduationCap, MapPin, Star, Upload, FileText, X } from "lucide-react";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { useMemo, useState, useRef } from "react";

interface StudentProfile {
  id: string;
  name: string;
  email: string;
  university: string;
  major: string;
  graduationYear: number;
  gpa: string;
  skills: string[];
  interests: string[];
  location: string;
  resume?: string; // URL or file path
  resumeFileName?: string;
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

interface RecommendedInternship {
  id: string;
  title: string;
  company: string;
  location: string;
  duration: string;
  stipend: string;
  skills: string[];
  matchScore: number;
  matchReasons: string[];
  status: 'open' | 'closed';
  aiAnalysis?: AIAnalysis;
}

interface StudentDashboardProps {
  profile: StudentProfile;
  recommendedInternships: RecommendedInternship[];
  applications: any[];
  onApplyToInternship?: (internshipId: string) => void;
  onUpdateProfile?: () => void;
  onSaveProfile?: (payload: Partial<StudentProfile> & {
    skills?: string[];
    interests?: string[];
    resume?: string;
    resumeFile?: File;
  }) => Promise<void> | void;
}

export default function StudentDashboard({
  profile,
  recommendedInternships,
  applications,
  onApplyToInternship,
  onUpdateProfile,
  onSaveProfile
}: StudentDashboardProps) {
  const [isEditOpen, setIsEditOpen] = useState(false);
  const [resumeFile, setResumeFile] = useState<File | null>(null);
  const [resumeUploading, setResumeUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  const [form, setForm] = useState(() => ({
    name: profile.name || "",
    university: profile.university || "",
    major: profile.major || "",
    graduationYear: profile.graduationYear || new Date().getFullYear(),
    gpa: profile.gpa || "",
    skills: (profile.skills || []).join(", "),
    interests: (profile.interests || []).join(", "),
    location: profile.location || "",
  }));

  const canSave = useMemo(() => {
    return form.name.trim().length >= 2 && form.university.trim() && form.major.trim();
  }, [form]);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      // Validate file type
      const allowedTypes = ['application/pdf', 'application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'];
      if (!allowedTypes.includes(file.type)) {
        alert('Please upload only PDF, DOC, or DOCX files');
        return;
      }
      
      // Validate file size (5MB limit)
      if (file.size > 5 * 1024 * 1024) {
        alert('File size must be less than 5MB');
        return;
      }
      
      setResumeFile(file);
    }
  };

  const handleRemoveFile = () => {
    setResumeFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const triggerFileUpload = () => {
    fileInputRef.current?.click();
  };

  const activeApplications = applications.filter(app => app.status === 'pending').length;
  const acceptedApplications = applications.filter(app => app.status === 'accepted').length;
  const averageMatchScore = recommendedInternships.length > 0 
    ? Math.round(recommendedInternships.reduce((sum, intern) => sum + intern.matchScore, 0) / recommendedInternships.length)
    : 0;

  return (
    <div className="space-y-6 p-6" data-testid="dashboard-student">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Welcome back, {profile.name.split(' ')[0]}!</h1>
          <p className="text-muted-foreground">Find your perfect PM internship match</p>
        </div>
        <Button onClick={() => setIsEditOpen(true)} variant="outline" data-testid="button-update-profile">
          <User className="h-4 w-4 mr-2" />
          Update Profile
        </Button>
      </div>

      {/* Metrics Row */}
      <div className="grid gap-4 md:grid-cols-4">
        <MetricsCard
          title="Active Applications"
          value={activeApplications}
          description="In review"
          icon={<Search className="h-4 w-4" />}
        />
        <MetricsCard
          title="Offers Received"
          value={acceptedApplications}
          description="Congratulations!"
          icon={<Star className="h-4 w-4" />}
        />
        <MetricsCard
          title="Avg Match Score"
          value={`${averageMatchScore}%`}
          description="AI compatibility"
          icon={<GraduationCap className="h-4 w-4" />}
        />
        <MetricsCard
          title="Profile Strength"
          value="85%"
          description="Complete your profile"
          progress={{ value: 85, max: 100, label: "Completeness" }}
          icon={<User className="h-4 w-4" />}
        />
      </div>

      <div className="grid gap-6 md:grid-cols-3">
        {/* Left Column - Profile & Applications */}
        <div className="space-y-6">
          {/* Profile Summary */}
          <Card data-testid="card-profile-summary">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <User className="h-5 w-5" />
                Profile Summary
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center gap-3">
                <Avatar className="h-12 w-12">
                  <AvatarImage src="" />
                  <AvatarFallback className="bg-primary text-primary-foreground">
                    {profile.name.split(' ').map(n => n[0]).join('')}
                  </AvatarFallback>
                </Avatar>
                <div>
                  <h3 className="font-semibold">{profile.name}</h3>
                  <p className="text-sm text-muted-foreground">{profile.email}</p>
                </div>
              </div>

              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <GraduationCap className="h-4 w-4 text-muted-foreground" />
                  <span className="text-sm">{profile.major} • {profile.university}</span>
                </div>
                <div className="flex items-center gap-2">
                  <MapPin className="h-4 w-4 text-muted-foreground" />
                  <span className="text-sm">{profile.location}</span>
                </div>
              </div>

              <div>
                <p className="text-sm font-medium mb-2">Skills</p>
                <div className="flex flex-wrap gap-1">
                  {profile.skills.slice(0, 6).map((skill, index) => (
                    <Badge key={index} variant="secondary" className="text-xs">
                      {skill}
                    </Badge>
                  ))}
                  {profile.skills.length > 6 && (
                    <Badge variant="outline" className="text-xs">
                      +{profile.skills.length - 6}
                    </Badge>
                  )}
                </div>
              </div>

              {profile.resume && (
                <div>
                  <p className="text-sm font-medium mb-2">Resume</p>
                  <div className="flex items-center gap-2 text-sm text-muted-foreground">
                    <FileText className="h-4 w-4" />
                    <span>{profile.resumeFileName || 'Resume uploaded'}</span>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Application Tracker */}
          <ApplicationTracker
            applications={applications}
            onViewDetails={(id) => console.log('View application:', id)}
            onWithdraw={(id) => console.log('Withdraw application:', id)}
          />
        </div>

        {/* Right Columns - Recommended Internships */}
        <div className="md:col-span-2 space-y-6">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-semibold">AI-Recommended Internships</h2>
            <div className="flex items-center gap-2">
              <Input 
                placeholder="Search internships..." 
                className="w-64"
                data-testid="input-search-internships"
              />
              <Button variant="outline" size="icon" data-testid="button-search">
                <Search className="h-4 w-4" />
              </Button>
            </div>
          </div>

          <div className="grid gap-4">
            {recommendedInternships.map((internship) => (
              <MatchCard
                key={internship.id}
                id={internship.id}
                type="internship"
                title={internship.title}
                company={internship.company}
                location={internship.location}
                matchScore={internship.matchScore}
                matchReasons={internship.matchReasons}
                duration={internship.duration}
                stipend={internship.stipend}
                skills={internship.skills}
                status={internship.status}
                onApply={() => onApplyToInternship?.(internship.id)}
                aiAnalysis={internship.aiAnalysis}
              />
            ))}
          </div>

          {recommendedInternships.length === 0 && (
            <Card>
              <CardContent className="py-8 text-center">
                <p className="text-muted-foreground">No internships found</p>
                <p className="text-sm text-muted-foreground mt-1">
                  Complete your profile to get better recommendations
                </p>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
      <Dialog open={isEditOpen} onOpenChange={setIsEditOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Edit Profile</DialogTitle>
          </DialogHeader>
          <div className="grid gap-4 py-2">
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="name" className="text-right">Name</Label>
              <Input id="name" className="col-span-3" value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} />
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="university" className="text-right">University</Label>
              <Input id="university" className="col-span-3" value={form.university} onChange={(e) => setForm({ ...form, university: e.target.value })} />
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="major" className="text-right">Major</Label>
              <Input id="major" className="col-span-3" value={form.major} onChange={(e) => setForm({ ...form, major: e.target.value })} />
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="graduationYear" className="text-right">Graduation Year</Label>
              <Input id="graduationYear" type="number" className="col-span-3" value={form.graduationYear} onChange={(e) => setForm({ ...form, graduationYear: Number(e.target.value) })} />
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="gpa" className="text-right">GPA</Label>
              <Input id="gpa" className="col-span-3" value={form.gpa} onChange={(e) => setForm({ ...form, gpa: e.target.value })} />
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="skills" className="text-right">Skills</Label>
              <Input id="skills" placeholder="comma separated" className="col-span-3" value={form.skills} onChange={(e) => setForm({ ...form, skills: e.target.value })} />
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="interests" className="text-right">Interests</Label>
              <Input id="interests" placeholder="comma separated" className="col-span-3" value={form.interests} onChange={(e) => setForm({ ...form, interests: e.target.value })} />
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="location" className="text-right">Location</Label>
              <Input id="location" className="col-span-3" value={form.location} onChange={(e) => setForm({ ...form, location: e.target.value })} />
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="resume" className="text-right">Resume</Label>
              <div className="col-span-3 space-y-2">
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".pdf,.doc,.docx"
                  onChange={handleFileChange}
                  className="hidden"
                />
                
                {resumeFile ? (
                  <div className="flex items-center gap-2 p-2 border rounded-md bg-muted/50">
                    <FileText className="h-4 w-4 text-green-600" />
                    <span className="text-sm flex-1">{resumeFile.name}</span>
                    <span className="text-xs text-muted-foreground">
                      {(resumeFile.size / 1024 / 1024).toFixed(1)}MB
                    </span>
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      onClick={handleRemoveFile}
                      className="h-6 w-6 p-0"
                    >
                      <X className="h-3 w-3" />
                    </Button>
                  </div>
                ) : profile.resume ? (
                  <div className="flex items-center gap-2 p-2 border rounded-md">
                    <FileText className="h-4 w-4 text-blue-600" />
                    <span className="text-sm flex-1">{profile.resumeFileName || 'Current resume'}</span>
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      onClick={triggerFileUpload}
                    >
                      Replace
                    </Button>
                  </div>
                ) : (
                  <Button
                    type="button"
                    variant="outline"
                    onClick={triggerFileUpload}
                    className="w-full"
                    disabled={resumeUploading}
                  >
                    <Upload className="h-4 w-4 mr-2" />
                    {resumeUploading ? 'Uploading...' : 'Upload Resume'}
                  </Button>
                )}
                
                <p className="text-xs text-muted-foreground">
                  Supported formats: PDF, DOC, DOCX (Max 5MB)
                </p>
              </div>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setIsEditOpen(false)}>Cancel</Button>
            <Button
              onClick={async () => {
                try {
                  setResumeUploading(true);
                  const payload = {
                    name: form.name,
                    university: form.university,
                    major: form.major,
                    graduationYear: form.graduationYear,
                    gpa: form.gpa || undefined,
                    skills: form.skills.split(',').map(s => s.trim()).filter(Boolean),
                    interests: form.interests.split(',').map(s => s.trim()).filter(Boolean),
                    location: form.location || undefined,
                    resumeFile: resumeFile || undefined,
                  };
                  await onSaveProfile?.(payload);
                  setIsEditOpen(false);
                  setResumeFile(null);
                } catch (error) {
                  console.error('Error saving profile:', error);
                } finally {
                  setResumeUploading(false);
                }
              }}
              disabled={!canSave || resumeUploading}
            >
              {resumeUploading ? 'Saving...' : 'Save'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}