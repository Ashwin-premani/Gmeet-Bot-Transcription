import type { Express, Request, Response } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import bcrypt from "bcrypt";
import { insertUserSchema, insertStudentSchema, insertEmployerSchema, UserRole } from "@shared/schema";
import { z } from "zod";
import { upload, processUploadedFile, deleteFile } from "./fileProcessor";

// Custom request interface for authenticated routes
interface AuthRequest extends Request {
  user?: {
    id: string;
    email: string;
    role: string;
  };
}

// Validation schemas
const loginSchema = z.object({
  email: z.string().email(),
  password: z.string().min(6),
  userType: z.enum(["student", "employer"])
});

const studentRegistrationSchema = z.object({
  name: z.string().min(2),
  email: z.string().email(),
  password: z.string().min(6),
  university: z.string().min(2),
  major: z.string().min(2),
  graduationYear: z.number().min(2024).max(2030),
  gpa: z.string().optional(),
  skills: z.string(),
  interests: z.string()
});

const employerRegistrationSchema = z.object({
  name: z.string().min(2),
  email: z.string().email(),
  password: z.string().min(6),
  companyName: z.string().min(2),
  industry: z.string().min(2),
  companySize: z.string().optional(),
  location: z.string().min(2),
  description: z.string().optional(),
  website: z.string().optional()
});

// Utility functions
const hashPassword = async (password: string): Promise<string> => {
  return await bcrypt.hash(password, 10);
};

const verifyPassword = async (password: string, hash: string): Promise<boolean> => {
  return await bcrypt.compare(password, hash);
};

export async function registerRoutes(app: Express): Promise<Server> {
  // Auth routes
  app.post("/api/auth/login", async (req: Request, res: Response) => {
    try {
      const { email, password, userType } = loginSchema.parse(req.body);
      
      // Find user by email
      const user = await storage.getUserByEmail(email);
      if (!user) {
        return res.status(401).json({ error: "Invalid credentials" });
      }
      
      // Check if user type matches
      if (user.role !== userType) {
        return res.status(401).json({ error: "Invalid user type for this account" });
      }
      
      // Verify password
      const isValidPassword = await verifyPassword(password, user.password);
      if (!isValidPassword) {
        return res.status(401).json({ error: "Invalid credentials" });
      }
      
      // Store user session
      req.session.user = {
        id: user.id,
        email: user.email,
        role: user.role
      };
      
      res.json({
        success: true,
        user: {
          id: user.id,
          name: user.name,
          email: user.email,
          role: user.role
        },
        profile: user.student || user.employer || null
      });
    } catch (error) {
      console.error('Login error:', error);
      res.status(400).json({ error: "Invalid request data" });
    }
  });
  
  app.post("/api/auth/register/student", async (req: Request, res: Response) => {
    try {
      const data = studentRegistrationSchema.parse(req.body);
      
      // Check if user already exists
      const existingUser = await storage.getUserByEmail(data.email);
      if (existingUser) {
        return res.status(400).json({ error: "User already exists with this email" });
      }
      
      // Hash password
      const hashedPassword = await hashPassword(data.password);
      
      // Create user
      const user = await storage.createUser({
        name: data.name,
        email: data.email,
        password: hashedPassword,
        role: UserRole.STUDENT
      });
      
      // Create student profile
      const student = await storage.createStudent({
        userId: user.id,
        university: data.university,
        major: data.major,
        graduationYear: data.graduationYear,
        gpa: data.gpa,
        skills: data.skills.split(',').map(s => s.trim()),
        interests: data.interests.split(',').map(s => s.trim()),
        preferences: {}
      });
      
      // Store user session
      req.session.user = {
        id: user.id,
        email: user.email,
        role: user.role
      };
      
      res.json({
        success: true,
        user: {
          id: user.id,
          name: user.name,
          email: user.email,
          role: user.role
        },
        profile: student
      });
    } catch (error) {
      console.error('Student registration error:', error);
      res.status(400).json({ error: error instanceof Error ? error.message : "Registration failed" });
    }
  });
  
  app.post("/api/auth/register/employer", async (req: Request, res: Response) => {
    try {
      const data = employerRegistrationSchema.parse(req.body);
      
      // Check if user already exists
      const existingUser = await storage.getUserByEmail(data.email);
      if (existingUser) {
        return res.status(400).json({ error: "User already exists with this email" });
      }
      
      // Hash password
      const hashedPassword = await hashPassword(data.password);
      
      // Create user
      const user = await storage.createUser({
        name: data.name,
        email: data.email,
        password: hashedPassword,
        role: UserRole.EMPLOYER
      });
      
      // Create employer profile
      const employer = await storage.createEmployer({
        userId: user.id,
        companyName: data.companyName,
        industry: data.industry,
        companySize: data.companySize || '',
        location: data.location,
        description: data.description,
        website: data.website
      });
      
      // Store user session
      req.session.user = {
        id: user.id,
        email: user.email,
        role: user.role
      };
      
      res.json({
        success: true,
        user: {
          id: user.id,
          name: user.name,
          email: user.email,
          role: user.role
        },
        profile: employer
      });
    } catch (error) {
      console.error('Employer registration error:', error);
      res.status(400).json({ error: error instanceof Error ? error.message : "Registration failed" });
    }
  });
  
  app.post("/api/auth/logout", async (req: Request, res: Response) => {
    req.session.destroy((err: any) => {
      if (err) {
        console.error('Logout error:', err);
        return res.status(500).json({ error: "Logout failed" });
      }
      res.json({ success: true });
    });
  });
  
  app.get("/api/auth/me", async (req: Request, res: Response) => {
    const user = req.session.user;
    if (!user) {
      return res.status(401).json({ error: "Not authenticated" });
    }
    
    try {
      const fullUser = await storage.getUserByEmail(user.email);
      if (!fullUser) {
        return res.status(401).json({ error: "User not found" });
      }
      
      res.json({
        user: {
          id: fullUser.id,
          name: fullUser.name,
          email: fullUser.email,
          role: fullUser.role
        },
        profile: fullUser.student || fullUser.employer || null
      });
    } catch (error) {
      console.error('Me endpoint error:', error);
      res.status(500).json({ error: "Server error" });
    }
  });

  // Employer data routes
  app.get("/api/employer/internships", async (req: Request, res: Response) => {
    try {
      const user = req.session.user;
      console.log("user", user);
      console.log({session: req.session});
      
      if (!user || user.role !== "employer") {
        return res.status(401).json({ error: "Not authenticated as employer" });
      }
      // employer profile id equals users.id in schema? We store employer by userId
      const employer = await storage.getEmployer(user.id);
      if (!employer) return res.json([]);
      const internships = await storage.getInternships({ employerId: employer.id });
      res.json(internships);
    } catch (error) {
      console.error('Employer internships error:', error);
      res.status(500).json({ error: "Server error" });
    }
  });

  app.get("/api/employer/recommended-candidates", async (req: Request, res: Response) => {
    try {
      const user = req.session.user;
      if (!user || user.role !== "employer") {
        return res.status(401).json({ error: "Not authenticated as employer" });
      }
      const employer = await storage.getEmployer(user.id);
      if (!employer) return res.json([]);
      // For now, gather recommendations for each internship owned by employer and flatten
      const internships = await storage.getInternships({ employerId: employer.id });
      const results = [] as any[];
      for (const internship of internships) {
        const recs = await storage.getRecommendedCandidates(internship.id, 10);
        results.push(...recs.map(r => ({ ...r, internshipId: internship.id })));
      }
      res.json(results);
    } catch (error) {
      console.error('Employer recommended candidates error:', error);
      res.status(500).json({ error: "Server error" });
    }
  });

  const createInternshipSchema = z.object({
    title: z.string().min(1),
    description: z.string().min(1),
    location: z.string().min(1),
    duration: z.string().min(1),
    stipend: z.string().optional(),
    requirements: z.string().min(1),
    skills: z.array(z.string()),
    applicationDeadline: z.string(),
    startDate: z.string(),
    maxApplications: z.number().min(1),
    status: z.enum(["open", "closed"]).default("open"),
  });

  app.post("/api/employer/internships", async (req: Request, res: Response) => {
    try {
      console.log("create internship");
      console.log({session: req.session});
      const user = req.session.user;
      if (!user || user.role !== "employer") {
        return res.status(401).json({ error: "Not authenticated as employer" });
      }
      const employer = await storage.getEmployer(user.id);
      if (!employer) {
        return res.status(404).json({ error: "Employer profile not found" });
      }
      
      const data = createInternshipSchema.parse(req.body);
      const internship = await storage.createInternship({
        employerId: employer.id,
        ...data,
        requirements: data.requirements.split('\n').filter(r => r.trim()),
      });
      
      res.json(internship);
    } catch (error) {
      console.error('Create internship error:', error);
      res.status(400).json({ error: error instanceof Error ? error.message : "Failed to create internship" });
    }
  });

  // Get internship details for employer
  app.get("/api/employer/internships/:id", async (req: Request, res: Response) => {
    try {
      const user = req.session.user;
      if (!user || user.role !== "employer") {
        return res.status(401).json({ error: "Not authenticated as employer" });
      }

      const { id } = req.params;
      const employer = await storage.getEmployer(user.id);
      if (!employer) {
        return res.status(404).json({ error: "Employer profile not found" });
      }

      // Get internship details with applications
      const internship = await storage.getInternshipById(id);
      if (!internship) {
        return res.status(404).json({ error: "Internship not found" });
      }

      // Verify this internship belongs to the employer
      if (internship.employerId !== employer.id) {
        return res.status(403).json({ error: "Access denied" });
      }

      // Get applications for this internship
      const applications = await storage.getApplications({ internshipId: id });
      
      // Get employer info
      const employerInfo = await storage.getEmployerById(employer.id);

      res.json({
        ...internship,
        applications,
        employer: employerInfo
      });
    } catch (error) {
      console.error('Get internship details error:', error);
      res.status(500).json({ error: "Server error" });
    }
  });

  // Student data routes
  app.get("/api/student/recommended-internships", async (req: Request, res: Response) => {
    try {
      const user = req.session.user;
      if (!user || user.role !== "student") {
        return res.status(401).json({ error: "Not authenticated as student" });
      }
      console.log("Looking for student with user.id:", user.id);
      const student = await storage.getStudent(user.id);
      if (!student) return res.json([]);
      console.log("student", student);
      
      const recommendations = await storage.getRecommendedInternships(user.id, 20);
      res.json(recommendations);
    } catch (error) {
      console.error('Student recommended internships error:', error);
      res.status(500).json({ error: "Server error" });
    }
  });

  app.get("/api/student/applications", async (req: Request, res: Response) => {
    try {
      const user = req.session.user;
      if (!user || user.role !== "student") {
        return res.status(401).json({ error: "Not authenticated as student" });
      }
      const student = await storage.getStudent(user.id);
      if (!student) return res.json([]);
      
      const applications = await storage.getApplications({ studentId: student.id });
      res.json(applications);
    } catch (error) {
      console.error('Student applications error:', error);
      res.status(500).json({ error: "Server error" });
    }
  });

  // Update student profile
  const updateStudentProfileSchema = z.object({
    name: z.string().min(2).optional(),
    university: z.string().min(1).optional(),
    major: z.string().min(1).optional(),
    graduationYear: z.number().min(2024).max(2035).optional(),
    gpa: z.string().optional(),
    skills: z.array(z.string()).optional(),
    interests: z.array(z.string()).optional(),
    location: z.string().optional(),
    resume: z.string().url().optional(),
  });

  // Resume upload endpoint
  app.post("/api/student/upload-resume", upload.single('resume'), async (req: Request, res: Response) => {
    try {
      const user = req.session.user;
      if (!user || user.role !== "student") {
        return res.status(401).json({ error: "Not authenticated as student" });
      }

      if (!req.file) {
        return res.status(400).json({ error: "No file uploaded" });
      }

      // Process the uploaded file
      const processedFile = await processUploadedFile(req.file);
      
      // Update student record with resume information
      const student = await storage.getStudent(user.id);
      if (!student) {
        // Clean up uploaded file if student not found
        await deleteFile(processedFile.path);
        return res.status(404).json({ error: "Student profile not found" });
      }

      // Clean up old resume file if exists
      if (student.resume) {
        await deleteFile(student.resume);
      }

      // Update student with new resume info
      const updatedStudent = await storage.updateStudent(user.id, {
        resume: processedFile.path,
        resumeFileName: processedFile.originalName,
        resumeText: processedFile.sanitizedText
      });

      res.json({
        success: true,
        resume: {
          filename: processedFile.originalName,
          size: processedFile.size,
          uploadedAt: new Date().toISOString()
        }
      });
    } catch (error) {
      console.error('Resume upload error:', error);
      
      // Clean up file on error
      if (req.file) {
        await deleteFile(req.file.path);
      }
      
      res.status(500).json({ 
        error: error instanceof Error ? error.message : "Failed to upload resume" 
      });
    }
  });

  app.put("/api/student/profile", async (req: Request, res: Response) => {
    try {
      const user = (req as AuthRequest).user || req.session.user;
      if (!user || user.role !== "student") {
        return res.status(401).json({ error: "Not authenticated as student" });
      }

      const data = updateStudentProfileSchema.parse(req.body);

      // Update user table if name provided
      if (data.name) {
        await storage.updateUser(user.id, { name: data.name });
      }

      // Update student table
      const studentUpdates: any = {};
      if (data.university !== undefined) studentUpdates.university = data.university;
      if (data.major !== undefined) studentUpdates.major = data.major;
      if (data.graduationYear !== undefined) studentUpdates.graduationYear = data.graduationYear;
      if (data.gpa !== undefined) studentUpdates.gpa = data.gpa;
      if (data.skills !== undefined) studentUpdates.skills = data.skills;
      if (data.interests !== undefined) studentUpdates.interests = data.interests;
      if (data.location !== undefined) studentUpdates.location = data.location;
      if (data.resume !== undefined) studentUpdates.resume = data.resume;

      const updated = await storage.updateStudent(user.id, studentUpdates);

      // Return fresh combined user + profile
      const fullUser = await storage.getUserByEmail((req.session.user as any).email);
      res.json({
        success: true,
        user: {
          id: fullUser!.id,
          name: fullUser!.name,
          email: fullUser!.email,
          role: fullUser!.role,
        },
        profile: updated,
      });
    } catch (error) {
      console.error('Update student profile error:', error);
      res.status(400).json({ error: error instanceof Error ? error.message : "Failed to update profile" });
    }
  });

  // Test AI matching endpoint
  app.post("/api/test/ai-match", async (req: Request, res: Response) => {
    try {
      const { student, internship } = req.body;
      
      // Import and test AI service
      const { aiService } = await import("./aiService");
      
      const analysis = await aiService.analyzeMatch({ student, internship });
      
      res.json({
        success: true,
        analysis
      });
    } catch (error) {
      console.error('AI test error:', error);
      res.status(500).json({ 
        error: error instanceof Error ? error.message : "AI test failed" 
      });
    }
  });

  const httpServer = createServer(app);

  return httpServer;
}
