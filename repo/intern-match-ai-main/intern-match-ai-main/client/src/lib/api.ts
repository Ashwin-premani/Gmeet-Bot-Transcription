import axios from "axios";

export const api = axios.create({
  baseURL: "/api",
  withCredentials: true,
  headers: {
    "Content-Type": "application/json",
  },
});

export interface AuthUser {
  id: string;
  name: string;
  email: string;
  role: "student" | "employer" | "admin";
}

export async function login(payload: { email: string; password: string; userType: "student" | "employer" }) {
  const { data } = await api.post("/auth/login", payload);
  return data as { success: boolean; user: AuthUser; profile: any };
}

export async function registerStudent(payload: {
  name: string;
  email: string;
  password: string;
  university: string;
  major: string;
  graduationYear: number;
  gpa?: string;
  skills: string;
  interests: string;
}) {
  const { data } = await api.post("/auth/register/student", payload);
  return data as { success: boolean; user: AuthUser; profile: any };
}

export async function registerEmployer(payload: {
  name: string;
  email: string;
  password: string;
  companyName: string;
  industry: string;
  companySize?: string;
  location: string;
  description?: string;
  website?: string;
}) {
  const { data } = await api.post("/auth/register/employer", payload);
  return data as { success: boolean; user: AuthUser; profile: any };
}

export async function me() {
  const { data } = await api.get("/auth/me");
  return data as { user: AuthUser; profile: any };
}

export async function logout() {
  const { data } = await api.post("/auth/logout");
  return data as { success: boolean };
}

export interface InternshipWithEmployer {
  id: string;
  title: string;
  location: string;
  duration?: string;
  stipend?: string;
  skills?: string[];
  status: string;
  employer: {
    id: string;
    companyName: string;
  };
  aiAnalysis?: any;
}

export async function getEmployerInternships() {
  const { data } = await api.get("/employer/internships");
  return data as InternshipWithEmployer[];
}

export interface RecommendedCandidate {
  id: string;
  name: string;
  user: AuthUser;
  university?: string;
  major?: string;
  location?: string;
  skills?: string[];
  matchScore: number;
  matchReasons: string[];
  internshipId: string;
  aiAnalysis?: any;
}

export async function getEmployerRecommendedCandidates() {
  const { data } = await api.get("/employer/recommended-candidates");
  console.log("getEmployerRecommendedCandidates", data);
  return data as RecommendedCandidate[];
}

export interface CreateInternshipPayload {
  title: string;
  description: string;
  location: string;
  duration: string;
  stipend?: string;
  requirements: string;
  skills: string[];
  applicationDeadline: string;
  startDate: string;
  maxApplications: number;
  status: "open" | "closed";
}

export async function createInternship(payload: CreateInternshipPayload) {
  const { data } = await api.post("/employer/internships", payload);
  return data;
}

export interface EmployerInternshipDetails {
  id: string;
  title: string;
  description: string;
  location: string;
  duration: string;
  stipend?: string;
  requirements: string[];
  skills: string[];
  status: string;
  maxApplications: number;
  currentApplications: number;
  createdAt: string;
  deadline: string;
  applications: Array<{
    id: string;
    status: string;
    appliedAt: string;
    aiMatchScore?: number;
    matchReasons?: string[];
    student: {
      id: string;
      university: string;
      major: string;
      skills: string[];
      gpa?: string;
      location: string;
    };
    internship: {
      id: string;
      title: string;
    };
  }>;
  employer: {
    id: string;
    companyName: string;
    industry: string;
    location: string;
    description?: string;
  };
}

export async function getEmployerInternshipDetails(id: string) {
  const { data } = await api.get(`/employer/internships/${id}`);
  return data as EmployerInternshipDetails;
}

// Student API functions
export interface StudentRecommendedInternship {
  id: string;
  title: string;
  location: string;
  duration?: string;
  stipend?: string;
  skills?: string[];
  matchScore: number;
  matchReasons: string[];
  status: string;
  employer: {
    id: string;
    companyName: string;
  };
  aiAnalysis?: any;
}

export interface StudentApplication {
  id: string;
  status: string;
  appliedDate?: string;
  internship: {
    id: string;
    title: string;
    location: string;
    employer: {
      companyName: string;
    };
  };
  aiAnalysis?: any;
}

export async function getStudentRecommendedInternships() {
  const { data } = await api.get("/student/recommended-internships");
  console.log("getStudentRecommendedInternships", data);
  return data as StudentRecommendedInternship[];
}

export async function getStudentApplications() {
  const { data } = await api.get("/student/applications");
  return data as StudentApplication[];
}

export interface UpdateStudentProfilePayload {
  name?: string;
  university?: string;
  major?: string;
  graduationYear?: number;
  gpa?: string;
  skills?: string[];
  interests?: string[];
  location?: string;
  resume?: string;
}

export async function updateStudentProfile(payload: UpdateStudentProfilePayload) {
  const { data } = await api.put("/student/profile", payload);
  return data as { success: boolean; user: AuthUser; profile: any };
}

export async function uploadResume(file: File) {
  const formData = new FormData();
  formData.append('resume', file);
  
  const { data } = await api.post("/student/upload-resume", formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return data as { success: boolean; resume: { filename: string; size: number; uploadedAt: string } };
}


