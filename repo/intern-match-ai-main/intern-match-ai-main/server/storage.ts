import { users, students, employers, internships, applications, aiMatches, type User, type Student, type Employer, type Internship, type Application, type InsertUser, type InsertStudent, type InsertEmployer, type InsertInternship, type InsertApplication, type UserWithProfile, type InternshipWithEmployer, type ApplicationWithDetails } from "@shared/schema";
import { randomUUID } from "crypto";
import { db } from "./db";
import { eq, and, desc } from "drizzle-orm";
import { aiService, type DeepSeekMatchRequest } from "./aiService";

// Interface for all storage operations
export interface IStorage {
  // User operations
  getUser(id: string): Promise<User | undefined>;
  getUserByEmail(email: string): Promise<UserWithProfile | undefined>;
  createUser(user: InsertUser): Promise<User>;
  updateUser(id: string, updates: Partial<InsertUser>): Promise<User>;
  
  // Student operations
  getStudent(userId: string): Promise<Student | undefined>;
  createStudent(student: InsertStudent): Promise<Student>;
  updateStudent(userId: string, updates: Partial<InsertStudent>): Promise<Student>;
  
  // Employer operations
  getEmployer(userId: string): Promise<Employer | undefined>;
  createEmployer(employer: InsertEmployer): Promise<Employer>;
  updateEmployer(userId: string, updates: Partial<InsertEmployer>): Promise<Employer>;
  
  // Internship operations
  createInternship(internship: InsertInternship): Promise<Internship>;
  getInternships(filters?: { status?: string; employerId?: string }): Promise<InternshipWithEmployer[]>;
  getInternship(id: string): Promise<InternshipWithEmployer | undefined>;
  getInternshipById(id: string): Promise<Internship | undefined>;
  updateInternship(id: string, updates: Partial<InsertInternship>): Promise<Internship>;
  
  // Employer operations extended
  getEmployerById(id: string): Promise<Employer | undefined>;
  
  // AI matching operations
  saveAIMatchAnalysis(studentId: string, internshipId: string, analysis: any): Promise<void>;
  getCachedAIAnalysis(studentId: string, internshipId: string): Promise<any | null>;
  invalidateStudentCache(studentId: string): Promise<void>;
  invalidateInternshipCache(internshipId: string): Promise<void>;
  
  // Application operations
  createApplication(application: InsertApplication): Promise<Application>;
  getApplications(filters?: { studentId?: string; internshipId?: string; status?: string }): Promise<ApplicationWithDetails[]>;
  updateApplication(id: string, updates: Partial<InsertApplication>): Promise<Application>;
  
  // AI matching operations
  getRecommendedInternships(studentId: string, limit?: number): Promise<(InternshipWithEmployer & { matchScore: number; matchReasons: string[] })[]>;
  getRecommendedCandidates(internshipId: string, limit?: number): Promise<(Student & { matchScore: number; matchReasons: string[] })[]>;
}

export class MemStorage implements IStorage {
  private users: Map<string, User>;

  constructor() {
    this.users = new Map();
  }

  async getUser(id: string): Promise<User | undefined> {
    return this.users.get(id);
  }

  async getUserByEmail(email: string): Promise<UserWithProfile | undefined> {
    return Array.from(this.users.values()).find(
      (user) => user.email === email,
    ) as UserWithProfile;
  }

  // Placeholder implementations for remaining methods
  async updateUser(id: string, updates: Partial<InsertUser>): Promise<User> {
    const user = this.users.get(id);
    if (!user) throw new Error('User not found');
    const updatedUser = { ...user, ...updates };
    this.users.set(id, updatedUser);
    return updatedUser;
  }

  async getStudent(userId: string): Promise<Student | undefined> {
    return undefined; // Placeholder
  }

  async createStudent(student: InsertStudent): Promise<Student> {
    throw new Error('Not implemented in memory storage');
  }

  async updateStudent(userId: string, updates: Partial<InsertStudent>): Promise<Student> {
    throw new Error('Not implemented in memory storage');
  }

  async getEmployer(userId: string): Promise<Employer | undefined> {
    return undefined; // Placeholder
  }

  async createEmployer(employer: InsertEmployer): Promise<Employer> {
    throw new Error('Not implemented in memory storage');
  }

  async updateEmployer(userId: string, updates: Partial<InsertEmployer>): Promise<Employer> {
    throw new Error('Not implemented in memory storage');
  }

  async createInternship(internship: InsertInternship): Promise<Internship> {
    throw new Error('Not implemented in memory storage');
  }

  async getInternships(filters?: { status?: string; employerId?: string }): Promise<InternshipWithEmployer[]> {
    return [];
  }

  async getInternship(id: string): Promise<InternshipWithEmployer | undefined> {
    return undefined;
  }

  async getInternshipById(id: string): Promise<Internship | undefined> {
    return undefined;
  }

  async getEmployerById(id: string): Promise<Employer | undefined> {
    return undefined;
  }

  async saveAIMatchAnalysis(studentId: string, internshipId: string, analysis: any): Promise<void> {
    // Not implemented in memory storage
    return;
  }

  async getCachedAIAnalysis(studentId: string, internshipId: string): Promise<any | null> {
    // Not implemented in memory storage
    return null;
  }

  async invalidateStudentCache(studentId: string): Promise<void> {
    // Not implemented in memory storage
    return;
  }

  async invalidateInternshipCache(internshipId: string): Promise<void> {
    // Not implemented in memory storage
    return;
  }

  async updateInternship(id: string, updates: Partial<InsertInternship>): Promise<Internship> {
    throw new Error('Not implemented in memory storage');
  }

  async createApplication(application: InsertApplication): Promise<Application> {
    throw new Error('Not implemented in memory storage');
  }

  async getApplications(filters?: { studentId?: string; internshipId?: string; status?: string }): Promise<ApplicationWithDetails[]> {
    return [];
  }

  async updateApplication(id: string, updates: Partial<InsertApplication>): Promise<Application> {
    throw new Error('Not implemented in memory storage');
  }

  async getRecommendedInternships(studentId: string, limit?: number): Promise<(InternshipWithEmployer & { matchScore: number; matchReasons: string[] })[]> {
    return [];
  }

  async getRecommendedCandidates(internshipId: string, limit?: number): Promise<(Student & { matchScore: number; matchReasons: string[] })[]> {
    return [];
  }

  async createUser(insertUser: InsertUser): Promise<User> {
    const id = randomUUID();
    const user: User = { ...insertUser, id, createdAt: new Date() };
    this.users.set(id, user);
    return user;
  }
}

// Database storage implementation
export class DatabaseStorage implements IStorage {
  // User operations
  async getUser(id: string): Promise<User | undefined> {
    const [user] = await db.select().from(users).where(eq(users.id, id));
    return user;
  }

  async getUserByEmail(email: string): Promise<UserWithProfile | undefined> {
    const [user] = await db.select().from(users).where(eq(users.email, email));
    if (!user) return undefined;

    let profile = {};
    if (user.role === 'student') {
      const [student] = await db.select().from(students).where(eq(students.userId, user.id));
      profile = { student };
    } else if (user.role === 'employer') {
      const [employer] = await db.select().from(employers).where(eq(employers.userId, user.id));
      profile = { employer };
    }

    return { ...user, ...profile };
  }

  async createUser(insertUser: InsertUser): Promise<User> {
    const [user] = await db.insert(users).values(insertUser).returning();
    return user;
  }

  async updateUser(id: string, updates: Partial<InsertUser>): Promise<User> {
    const [user] = await db.update(users).set(updates).where(eq(users.id, id)).returning();
    return user;
  }

  // Student operations
  async getStudent(userId: string): Promise<Student | undefined> {
    console.log("getStudent", {userId});
    const [student] = await db.select().from(students).where(eq(students.userId, userId));
    console.log("getStudent after query", {student});
    return student;
  }

  async createStudent(student: InsertStudent): Promise<Student> {
    const [newStudent] = await db.insert(students).values(student).returning();
    return newStudent;
  }

  async updateStudent(userId: string, updates: Partial<InsertStudent>): Promise<Student> {
    const [student] = await db.update(students).set(updates).where(eq(students.userId, userId)).returning();
    
    // Invalidate AI cache when student profile is updated
    await this.invalidateStudentCache(userId);
    
    return student;
  }

  // Employer operations
  async getEmployer(userId: string): Promise<Employer | undefined> {
    const [employer] = await db.select().from(employers).where(eq(employers.userId, userId));
    return employer;
  }

  async createEmployer(employer: InsertEmployer): Promise<Employer> {
    const [newEmployer] = await db.insert(employers).values(employer).returning();
    return newEmployer;
  }

  async updateEmployer(userId: string, updates: Partial<InsertEmployer>): Promise<Employer> {
    const [employer] = await db.update(employers).set(updates).where(eq(employers.userId, userId)).returning();
    return employer;
  }

  // Internship operations
  async createInternship(internship: InsertInternship): Promise<Internship> {
    const [newInternship] = await db.insert(internships).values(internship).returning();
    return newInternship;
  }

  async getInternships(filters?: { status?: string; employerId?: string }): Promise<InternshipWithEmployer[]> {
    const query = db.select().from(internships).leftJoin(employers, eq(internships.employerId, employers.id));
    
    if (filters?.status) {
      query.where(eq(internships.status, filters.status));
    }
    if (filters?.employerId) {
      query.where(eq(internships.employerId, filters.employerId));
    }

    const results = await query;
    return results.map(row => ({
      ...row.internships,
      employer: row.employers!
    }));
  }

  async getInternship(id: string): Promise<InternshipWithEmployer | undefined> {
    const [result] = await db.select()
      .from(internships)
      .leftJoin(employers, eq(internships.employerId, employers.id))
      .where(eq(internships.id, id));

    if (!result) return undefined;

    return {
      ...result.internships,
      employer: result.employers!
    };
  }

  async updateInternship(id: string, updates: Partial<InsertInternship>): Promise<Internship> {
    const [internship] = await db.update(internships).set(updates).where(eq(internships.id, id)).returning();
    
    // Invalidate AI cache when internship is updated
    await this.invalidateInternshipCache(id);
    
    return internship;
  }

  async getInternshipById(id: string): Promise<Internship | undefined> {
    const [internship] = await db.select().from(internships).where(eq(internships.id, id));
    return internship;
  }

  async getEmployerById(id: string): Promise<Employer | undefined> {
    const [employer] = await db.select().from(employers).where(eq(employers.id, id));
    return employer;
  }

  async saveAIMatchAnalysis(studentId: string, internshipId: string, analysis: any): Promise<void> {
    try {
      // Check if analysis already exists and update, otherwise insert
      const existingAnalysis = await this.getCachedAIAnalysis(studentId, internshipId);
      
      const analysisData = {
        studentId,
        internshipId,
        matchScore: analysis.overallMatch,
        confidence: analysis.confidence,
        keyStrengths: analysis.keyStrengths || [],
        potentialConcerns: analysis.potentialConcerns || [],
        skillGaps: analysis.skillGaps || [],
        careerImpact: analysis.careerImpact,
        employerBenefits: analysis.employerBenefits || [],
        actionableAdvice: analysis.actionableAdvice || [],
        skillsMatch: analysis.breakdown?.skillsMatch,
        experienceMatch: analysis.breakdown?.experienceMatch,
        locationMatch: analysis.breakdown?.locationMatch,
        cultureMatch: analysis.breakdown?.cultureMatch,
        careerFitMatch: analysis.breakdown?.careerFitMatch,
        reasons: analysis.keyStrengths || [], // Legacy field
      };

      if (existingAnalysis) {
        // Update existing record
        await db.update(aiMatches)
          .set(analysisData)
          .where(and(
            eq(aiMatches.studentId, studentId),
            eq(aiMatches.internshipId, internshipId)
          ));
      } else {
        // Insert new record
        await db.insert(aiMatches).values(analysisData);
      }
    } catch (error) {
      console.error('Error saving AI match analysis:', error);
      // Don't throw - this is supplementary data
    }
  }

  async getCachedAIAnalysis(studentId: string, internshipId: string): Promise<any | null> {
    try {
      const [cachedAnalysis] = await db.select()
        .from(aiMatches)
        .where(and(
          eq(aiMatches.studentId, studentId),
          eq(aiMatches.internshipId, internshipId)
        ))
        .orderBy(desc(aiMatches.createdAt))
        .limit(1);

      if (!cachedAnalysis) {
        return null;
      }

      // Convert database format back to analysis format
      return {
        overallMatch: cachedAnalysis.matchScore,
        confidence: cachedAnalysis.confidence || 80,
        keyStrengths: cachedAnalysis.keyStrengths || [],
        potentialConcerns: cachedAnalysis.potentialConcerns || [],
        skillGaps: cachedAnalysis.skillGaps || [],
        careerImpact: cachedAnalysis.careerImpact || '',
        employerBenefits: cachedAnalysis.employerBenefits || [],
        actionableAdvice: cachedAnalysis.actionableAdvice || [],
        breakdown: {
          skillsMatch: cachedAnalysis.skillsMatch || 50,
          experienceMatch: cachedAnalysis.experienceMatch || 50,
          locationMatch: cachedAnalysis.locationMatch || 50,
          cultureMatch: cachedAnalysis.cultureMatch || 50,
          careerFitMatch: cachedAnalysis.careerFitMatch || 50
        }
      };
    } catch (error) {
      console.error('Error retrieving cached AI analysis:', error);
      return null;
    }
  }

  async invalidateStudentCache(studentId: string): Promise<void> {
    try {
      // Delete all cached analyses for this student
      await db.delete(aiMatches).where(eq(aiMatches.studentId, studentId));
      console.log(`Invalidated AI cache for student ${studentId}`);
    } catch (error) {
      console.error('Error invalidating student cache:', error);
    }
  }

  async invalidateInternshipCache(internshipId: string): Promise<void> {
    try {
      // Delete all cached analyses for this internship
      await db.delete(aiMatches).where(eq(aiMatches.internshipId, internshipId));
      console.log(`Invalidated AI cache for internship ${internshipId}`);
    } catch (error) {
      console.error('Error invalidating internship cache:', error);
    }
  }

  // Application operations
  async createApplication(application: InsertApplication): Promise<Application> {
    const [newApplication] = await db.insert(applications).values(application).returning();
    return newApplication;
  }

  async getApplications(filters?: { studentId?: string; internshipId?: string; status?: string }): Promise<ApplicationWithDetails[]> {
    let query = db.select()
      .from(applications)
      .leftJoin(internships, eq(applications.internshipId, internships.id))
      .leftJoin(employers, eq(internships.employerId, employers.id))
      .leftJoin(students, eq(applications.studentId, students.id));

    const conditions = [];
    if (filters?.studentId) conditions.push(eq(applications.studentId, filters.studentId));
    if (filters?.internshipId) conditions.push(eq(applications.internshipId, filters.internshipId));
    if (filters?.status) conditions.push(eq(applications.status, filters.status));

    const results = conditions.length > 0 
      ? await query.where(and(...conditions))
      : await query;
    return results.map(row => ({
      ...row.applications,
      internship: {
        ...row.internships!,
        employer: row.employers!
      },
      student: row.students!
    }));
  }

  async updateApplication(id: string, updates: Partial<InsertApplication>): Promise<Application> {
    const [application] = await db.update(applications).set(updates).where(eq(applications.id, id)).returning();
    return application;
  }

  // AI matching operations with DeepSeek integration
  async getRecommendedInternships(studentId: string, limit = 10): Promise<(InternshipWithEmployer & { matchScore: number; matchReasons: string[] })[]> {
    // Get student profile
    const student = await this.getStudent(studentId);
    console.log("getRecommendedInternships", {student});
    if (!student) return [];

    // Get user info for student name
    const user = await this.getUser(studentId);
    if (!user) return [];

    // Get open internships
    const openInternships = await this.getInternships({ status: 'open' });
    
    // Use AI matching for each internship
    const aiMatches = await Promise.all(
      openInternships.map(async (internship) => {
        try {
          // Check if we have cached analysis first
          const cachedAnalysis = await this.getCachedAIAnalysis(studentId, internship.id);
          
          if (cachedAnalysis) {
            console.log(`Using cached AI analysis for student ${studentId} and internship ${internship.id}`);
            return {
              ...internship,
              matchScore: cachedAnalysis.overallMatch,
              matchReasons: cachedAnalysis.keyStrengths.slice(0, 4),
              aiAnalysis: cachedAnalysis
            };
          }

          console.log(`Generating new AI analysis for student ${studentId} and internship ${internship.id}`);
          
          const matchRequest: DeepSeekMatchRequest = {
            student: {
              name: user.name,
              university: student.university,
              major: student.major,
              graduationYear: student.graduationYear,
              gpa: student.gpa || undefined,
              skills: student.skills || [],
              interests: student.interests || [],
              location: student.location,
              resumeText: student.resumeText || undefined,
            },
            internship: {
              title: internship.title,
              description: internship.description,
              requirements: internship.requirements || [],
              skills: internship.skills || [],
              location: internship.location,
              duration: internship.duration,
              stipend: internship.stipend || undefined,
              company: {
                name: internship.employer.companyName,
                industry: internship.employer.industry,
                description: internship.employer.description || undefined,
              }
            }
          };

          const aiAnalysis = await aiService.analyzeMatch(matchRequest);
          
          // Save AI analysis to database for future use
          await this.saveAIMatchAnalysis(studentId, internship.id, aiAnalysis);
          
          return {
            ...internship,
            matchScore: aiAnalysis.overallMatch,
            matchReasons: aiAnalysis.keyStrengths.slice(0, 4),
            aiAnalysis // Include full analysis for UI
          };
        } catch (error) {
          console.error(`AI analysis failed for internship ${internship.id}:`, error);
          
          // Fallback to basic matching if AI fails
          return this.basicMatch(student, internship);
        }
      })
    );

    // Sort by match score and return top results
    return aiMatches
      .sort((a, b) => b.matchScore - a.matchScore)
      .slice(0, limit);
  }

  // Fallback basic matching algorithm
  private basicMatch(student: any, internship: any) {
    let score = 50;
    const reasons: string[] = [];

    // Skill matching
    if (student.skills && internship.skills) {
      const studentSkills = student.skills.map((s: string) => s.toLowerCase());
      const internshipSkills = internship.skills.map((s: string) => s.toLowerCase());
      const matchingSkills = studentSkills.filter((skill: string) => 
        internshipSkills.some((reqSkill: string) => reqSkill.includes(skill) || skill.includes(reqSkill))
      );
      
      if (matchingSkills.length > 0) {
        score += matchingSkills.length * 10;
        reasons.push(`Skills match: ${matchingSkills.slice(0, 3).join(', ')}`);
      }
    }

    // Graduation year consideration
    if (student.graduationYear && student.graduationYear <= 2025) {
      score += 15;
      reasons.push('Graduation timeline matches internship duration');
    }

    score = Math.min(100, Math.max(0, score));

    return {
      ...internship,
      matchScore: score,
      matchReasons: reasons.slice(0, 4)
    };
  }

  async getRecommendedCandidates(internshipId: string, limit = 10): Promise<(Student & { matchScore: number; matchReasons: string[] })[]> {
    // Get internship details
    const internship = await this.getInternship(internshipId);
    if (!internship) return [];

    // Get all students (simplified - in production would filter better)
    const allStudents = await db.select().from(students).leftJoin(users, eq(students.userId, users.id));
    console.log({allStudents});
    
    // Use AI matching for each candidate
    const aiMatches = await Promise.all(
      allStudents.map(async (studentRecord) => {
        try {
          const student = studentRecord.students;
          const user = studentRecord.users!;
          
          // Check if we have cached analysis first
          const cachedAnalysis = await this.getCachedAIAnalysis(student.userId, internshipId);
          
          if (cachedAnalysis) {
            console.log(`Using cached AI analysis for student ${student.userId} and internship ${internshipId}`);
            return {
              ...student,
              user,
              matchScore: cachedAnalysis.overallMatch,
              matchReasons: cachedAnalysis.keyStrengths.slice(0, 4),
              aiAnalysis: cachedAnalysis
            };
          }

          console.log(`Generating new AI analysis for student ${student.userId} and internship ${internshipId}`);
          
          const matchRequest: DeepSeekMatchRequest = {
            student: {
              name: user.name,
              university: student.university,
              major: student.major,
              graduationYear: student.graduationYear,
              gpa: student.gpa || undefined,
              skills: student.skills || [],
              interests: student.interests || [],
              location: student.location,
              resumeText: student.resumeText || undefined,
            },
            internship: {
              title: internship.title,
              description: internship.description,
              requirements: internship.requirements || [],
              skills: internship.skills || [],
              location: internship.location,
              duration: internship.duration,
              stipend: internship.stipend || undefined,
              company: {
                name: internship.employer.companyName,
                industry: internship.employer.industry,
                description: internship.employer.description || undefined,
              }
            }
          };

          const aiAnalysis = await aiService.analyzeMatch(matchRequest);
          
          // Save AI analysis to database for future use
          await this.saveAIMatchAnalysis(student.userId, internshipId, aiAnalysis);
          
          return {
            ...student,
            user,
            matchScore: aiAnalysis.overallMatch,
            matchReasons: aiAnalysis.keyStrengths.slice(0, 4),
            aiAnalysis // Include full analysis for UI
          };
        } catch (error) {
          console.error(`AI analysis failed for student ${studentRecord.users?.name}:`, error);
          
          // Fallback to basic matching if AI fails
          return this.basicCandidateMatch(studentRecord, internship);
        }
      })
    );

    return aiMatches
      .sort((a, b) => b.matchScore - a.matchScore)
      .slice(0, limit);
  }

  // Fallback basic candidate matching
  private basicCandidateMatch(studentRecord: any, internship: any) {
    const student = studentRecord.students;
    const user = studentRecord.users!;
    let score = 50;
    const reasons: string[] = [];

    // Skill matching
    if (student.skills && internship.skills) {
      const studentSkills = student.skills.map((s: string) => s.toLowerCase());
      const internshipSkills = internship.skills.map((s: string) => s.toLowerCase());
      const matchingSkills = studentSkills.filter((skill: string) => 
        internshipSkills.some((reqSkill: string) => reqSkill.includes(skill) || skill.includes(reqSkill))
      );
      
      if (matchingSkills.length > 0) {
        score += matchingSkills.length * 12;
        reasons.push(`Strong skills match: ${matchingSkills.slice(0, 3).join(', ')}`);
      }
    }

    // GPA consideration
    if (student.gpa) {
      const gpaNum = parseFloat(student.gpa);
      if (gpaNum >= 3.5) {
        score += 20;
        reasons.push('High academic performance');
      } else if (gpaNum >= 3.0) {
        score += 10;
        reasons.push('Good academic performance');
      }
    }

    score = Math.min(100, Math.max(0, score));
    
    return {
      ...student,
      user,
      matchScore: score,
      matchReasons: reasons.slice(0, 4)
    };
  }
}

export const storage = new DatabaseStorage();
