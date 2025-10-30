import { sql } from "drizzle-orm";
import { pgTable, text, varchar, integer, boolean, timestamp, jsonb } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

// User types enum for role-based access
export const UserRole = {
  STUDENT: 'student',
  EMPLOYER: 'employer',
  ADMIN: 'admin'
} as const;

export const ApplicationStatus = {
  PENDING: 'pending',
  ACCEPTED: 'accepted',
  REJECTED: 'rejected',
  WITHDRAWN: 'withdrawn'
} as const;

export const InternshipStatus = {
  OPEN: 'open',
  CLOSED: 'closed',
  FILLED: 'filled'
} as const;

// Users table
export const users = pgTable("users", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  email: text("email").notNull().unique(),
  password: text("password").notNull(),
  role: text("role").notNull(),
  name: text("name").notNull(),
  createdAt: timestamp("created_at").defaultNow(),
});

// Students table
export const students = pgTable("students", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").notNull(),
  university: text("university").notNull(),
  major: text("major").notNull(),
  graduationYear: integer("graduation_year").notNull(),
  skills: text("skills").array(),
  interests: text("interests").array(),
  gpa: text("gpa"),
  resume: text("resume_url"),
  resumeFileName: text("resume_file_name"),
  resumeText: text("resume_text"), // Extracted text content for AI processing
  location: text("location").notNull().default('India'),
  preferences: jsonb("preferences"),
});
// SQL QUERY TO ALTER THE TABLE TO ADD THE RESUME FILE NAME AND RESUME TEXT
// ALTER TABLE students ADD COLUMN resume_file_name TEXT;
// ALTER TABLE students ADD COLUMN resume_text TEXT;

// Employers table
export const employers = pgTable("employers", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").notNull(),
  companyName: text("company_name").notNull(),
  industry: text("industry").notNull(),
  companySize: text("company_size"),
  location: text("location").notNull(),
  description: text("description"),
  website: text("website"),
});

// Internships table
export const internships = pgTable("internships", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  employerId: varchar("employer_id").notNull(),
  title: text("title").notNull(),
  description: text("description").notNull(),
  requirements: text("requirements").array(),
  skills: text("skills").array(),
  location: text("location").notNull().default('India'),
  duration: text("duration").notNull(),
  stipend: text("stipend"),
  status: text("status").notNull().default('open'),
  maxApplications: integer("max_applications").default(100),
  currentApplications: integer("current_applications").default(0),
  createdAt: timestamp("created_at").defaultNow(),
  deadline: timestamp("deadline"),
});

// Applications table
export const applications = pgTable("applications", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  studentId: varchar("student_id").notNull(),
  internshipId: varchar("internship_id").notNull(),
  status: text("status").notNull().default('pending'),
  aiMatchScore: integer("ai_match_score"),
  matchReasons: text("match_reasons").array(),
  appliedAt: timestamp("applied_at").defaultNow(),
  reviewedAt: timestamp("reviewed_at"),
});

// AI Match data table
export const aiMatches = pgTable("ai_matches", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  studentId: varchar("student_id").notNull(),
  internshipId: varchar("internship_id").notNull(),
  matchScore: integer("match_score").notNull(),
  confidence: integer("confidence"),
  keyStrengths: text("key_strengths").array(),
  potentialConcerns: text("potential_concerns").array(),
  skillGaps: text("skill_gaps").array(),
  careerImpact: text("career_impact"),
  employerBenefits: text("employer_benefits").array(),
  actionableAdvice: text("actionable_advice").array(),
  // Breakdown scores
  skillsMatch: integer("skills_match"),
  experienceMatch: integer("experience_match"),
  locationMatch: integer("location_match"),
  cultureMatch: integer("culture_match"),
  careerFitMatch: integer("career_fit_match"),
  // Legacy fields
  interestsMatch: integer("interests_match"),
  reasons: text("reasons").array(),
  recommendationRank: integer("recommendation_rank"),
  createdAt: timestamp("created_at").defaultNow(),
});
// SQL QUERY TO ALTER THE TABLE TO ADD THE NEW FIELDS
// ALTER TABLE ai_matches ADD COLUMN confidence INTEGER;
// ALTER TABLE ai_matches ADD COLUMN key_strengths TEXT[];
// ALTER TABLE ai_matches ADD COLUMN potential_concerns TEXT[];
// ALTER TABLE ai_matches ADD COLUMN skill_gaps TEXT[];
// ALTER TABLE ai_matches ADD COLUMN career_impact TEXT;
// ALTER TABLE ai_matches ADD COLUMN employer_benefits TEXT[];
// ALTER TABLE ai_matches ADD COLUMN actionable_advice TEXT[];
// ALTER TABLE ai_matches ADD COLUMN experience_match INTEGER;
// ALTER TABLE ai_matches ADD COLUMN location_match INTEGER;
// ALTER TABLE ai_matches ADD COLUMN culture_match INTEGER;
// ALTER TABLE ai_matches ADD COLUMN career_fit_match INTEGER;
// ALTER TABLE ai_matches ADD COLUMN reasons TEXT[];
// ALTER TABLE ai_matches ADD COLUMN recommendation_rank INTEGER;

// Create insert schemas
export const insertUserSchema = createInsertSchema(users).omit({ id: true, createdAt: true });
export const insertStudentSchema = createInsertSchema(students).omit({ id: true });
export const insertEmployerSchema = createInsertSchema(employers).omit({ id: true });
export const insertInternshipSchema = createInsertSchema(internships).omit({ 
  id: true, 
  createdAt: true, 
  currentApplications: true 
});
export const insertApplicationSchema = createInsertSchema(applications).omit({ 
  id: true, 
  appliedAt: true, 
  reviewedAt: true 
});
export const insertAiMatchSchema = createInsertSchema(aiMatches).omit({ id: true, createdAt: true });

// Export types
export type User = typeof users.$inferSelect;
export type Student = typeof students.$inferSelect;
export type Employer = typeof employers.$inferSelect;
export type Internship = typeof internships.$inferSelect;
export type Application = typeof applications.$inferSelect;
export type AiMatch = typeof aiMatches.$inferSelect;

export type InsertUser = z.infer<typeof insertUserSchema>;
export type InsertStudent = z.infer<typeof insertStudentSchema>;
export type InsertEmployer = z.infer<typeof insertEmployerSchema>;
export type InsertInternship = z.infer<typeof insertInternshipSchema>;
export type InsertApplication = z.infer<typeof insertApplicationSchema>;
export type InsertAiMatch = z.infer<typeof insertAiMatchSchema>;

// Additional utility types
export type UserWithProfile = User & {
  student?: Student;
  employer?: Employer;
};

export type InternshipWithEmployer = Internship & {
  employer: Employer;
};

export type ApplicationWithDetails = Application & {
  internship: InternshipWithEmployer;
  student: Student;
};

export type MatchWithDetails = AiMatch & {
  internship: InternshipWithEmployer;
  student: Student;
};