import { useMemo } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import StudentDashboard from "./StudentDashboard";
import { getStudentRecommendedInternships, getStudentApplications, me, updateStudentProfile, uploadResume } from "@/lib/api";

export default function StudentDashboardContainer() {
  const { data: auth } = useQuery({ queryKey: ["auth", "me"], queryFn: me });
  const queryClient = useQueryClient();
  const { data: internships = [], isLoading: loadingInternships } = useQuery({
    queryKey: ["student", "recommended-internships"],
    queryFn: getStudentRecommendedInternships,
    enabled: !!auth?.user && auth.user.role === "student",
  });
  const { data: applications = [], isLoading: loadingApplications } = useQuery({
    queryKey: ["student", "applications"],
    queryFn: getStudentApplications,
    enabled: !!auth?.user && auth.user.role === "student",
  });

  const { mutateAsync: saveProfile, isPending: savingProfile } = useMutation({
    mutationFn: updateStudentProfile,
    onSuccess: async () => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["auth", "me"] }),
        queryClient.invalidateQueries({ queryKey: ["student", "recommended-internships"] }),
      ]);
    },
  });

  // Transform auth data to match StudentProfile interface
  const profile = useMemo(() => {
    if (!auth?.user || !auth?.profile) return null;
    
    return {
      id: auth.user.id,
      name: auth.user.name || "Student",
      email: auth.user.email,
      university: auth.profile.university || "",
      major: auth.profile.major || "",
      graduationYear: auth.profile.graduationYear || new Date().getFullYear(),
      gpa: auth.profile.gpa || "",
      skills: auth.profile.skills || [],
      interests: auth.profile.interests || [],
      location: auth.profile.location || "",
      resume: auth.profile.resume || undefined,
      resumeFileName: auth.profile.resumeFileName || undefined,
    };
  }, [auth]);

  // Transform internships to match expected format
  const recommendedInternships = useMemo(() => {
    const list = Array.isArray(internships) ? internships : [];
    return list.map((i) => ({
      id: i.id,
      title: i.title || "Internship",
      company: i.employer?.companyName || "Company",
      location: i.location || "",
      duration: i.duration || "",
      stipend: i.stipend || "",
      skills: i.skills || [],
      matchScore: i.matchScore,
      matchReasons: i.matchReasons || [],
      status: (i.status as "open" | "closed") || "open",
      aiAnalysis: i.aiAnalysis,
    }));
  }, [internships]);

  // Transform applications to match expected format
  const transformedApplications = useMemo(() => {
    const list = Array.isArray(applications) ? applications : [];
    return list.map((app) => ({
      id: app.id,
      internshipTitle: app.internship?.title || "Internship",
      company: app.internship?.employer?.companyName || "Company",
      location: app.internship?.location || "",
      appliedDate: app.appliedDate || new Date().toISOString(),
      status: app.status || "pending",
    }));
  }, [applications]);

  if (!profile) {
    return (
      <div className="p-6">
        <h1 className="text-2xl font-bold">Student Dashboard</h1>
        <p className="text-muted-foreground">Loading profile...</p>
      </div>
    );
  }

  return (
    <StudentDashboard
      profile={profile}
      recommendedInternships={recommendedInternships}
      applications={transformedApplications}
      onApplyToInternship={(id) => console.log('Apply to internship:', id)}
      onUpdateProfile={() => {}}
      onSaveProfile={async (payload) => {
        try {
          // Upload resume file first if provided
          if (payload.resumeFile) {
            await uploadResume(payload.resumeFile);
          }
          
          // Then update profile with other data
          await saveProfile({
            name: payload.name,
            university: payload.university,
            major: payload.major,
            graduationYear: payload.graduationYear,
            gpa: payload.gpa,
            skills: payload.skills,
            interests: payload.interests,
            location: payload.location,
            resume: payload.resume,
          });
        } catch (error) {
          console.error('Error saving profile:', error);
          throw error;
        }
      }}
    />
  );
}
