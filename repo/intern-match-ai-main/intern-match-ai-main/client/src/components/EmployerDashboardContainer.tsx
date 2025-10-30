import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { useLocation } from "wouter";
import EmployerDashboard from "./EmployerDashboard";
import { getEmployerInternships, getEmployerRecommendedCandidates, me } from "@/lib/api";

export default function EmployerDashboardContainer() {
  const [, setLocation] = useLocation();
  const { data: auth } = useQuery({ queryKey: ["auth", "me"], queryFn: me });
  const { data: internships = [], isLoading: loadingInternships } = useQuery({
    queryKey: ["employer", "internships"],
    queryFn: getEmployerInternships,
  });
  const { data: candidates = [], isLoading: loadingCandidates } = useQuery({
    queryKey: ["employer", "recommended-candidates"],
    queryFn: getEmployerRecommendedCandidates,
  });

  const companyName = auth?.profile?.companyName || auth?.user?.name || "Employer";

  const postedInternships = useMemo(() => {
    return internships.map((i) => ({
      id: i.id,
      title: i.title || "Untitled",
      location: i.location || "",
      duration: i.duration || "",
      applications: 0,
      maxApplications: 0,
      status: (i.status as "open" | "closed" | "filled") || "open",
      deadline: new Date().toISOString(),
    }));
  }, [internships]);

  const recommendedCandidates = useMemo(() => {
    return candidates.map((c) => ({
      id: c.id,
      name: c.user.name || "Candidate",
      university: c.university || "",
      major: c.major || "",
      location: c.location || "",
      skills: c.skills || [],
      matchScore: c.matchScore,
      matchReasons: c.matchReasons || [],
      status: "pending" as const,
      internshipId: c.internshipId,
      aiAnalysis: c.aiAnalysis,
    }));
  }, [candidates]);

  return (
    <EmployerDashboard
      companyName={companyName}
      postedInternships={postedInternships}
      recommendedCandidates={recommendedCandidates}
      onCreateInternship={() => setLocation("/post-internship")}
      onViewInternship={(internshipId) => setLocation(`/employer/internship/${internshipId}`)}
    />
  );
}


