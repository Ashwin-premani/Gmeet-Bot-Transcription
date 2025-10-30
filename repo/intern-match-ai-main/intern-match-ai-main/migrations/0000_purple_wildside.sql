CREATE TABLE "ai_matches" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"student_id" varchar NOT NULL,
	"internship_id" varchar NOT NULL,
	"match_score" integer NOT NULL,
	"skills_match" integer,
	"interests_match" integer,
	"location_match" integer,
	"reasons" text[],
	"recommendation_rank" integer,
	"created_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "applications" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"student_id" varchar NOT NULL,
	"internship_id" varchar NOT NULL,
	"status" text DEFAULT 'pending' NOT NULL,
	"ai_match_score" integer,
	"match_reasons" text[],
	"applied_at" timestamp DEFAULT now(),
	"reviewed_at" timestamp
);
--> statement-breakpoint
CREATE TABLE "employers" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"user_id" varchar NOT NULL,
	"company_name" text NOT NULL,
	"industry" text NOT NULL,
	"company_size" text,
	"location" text NOT NULL,
	"description" text,
	"website" text
);
--> statement-breakpoint
CREATE TABLE "internships" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"employer_id" varchar NOT NULL,
	"title" text NOT NULL,
	"description" text NOT NULL,
	"requirements" text[],
	"skills" text[],
	"location" text DEFAULT 'India' NOT NULL,
	"duration" text NOT NULL,
	"stipend" text,
	"status" text DEFAULT 'open' NOT NULL,
	"max_applications" integer DEFAULT 100,
	"current_applications" integer DEFAULT 0,
	"created_at" timestamp DEFAULT now(),
	"deadline" timestamp
);
--> statement-breakpoint
CREATE TABLE "students" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"user_id" varchar NOT NULL,
	"university" text NOT NULL,
	"major" text NOT NULL,
	"graduation_year" integer NOT NULL,
	"skills" text[],
	"interests" text[],
	"gpa" text,
	"resume_url" text,
	"location" text DEFAULT 'India' NOT NULL,
	"preferences" jsonb
);
--> statement-breakpoint
CREATE TABLE "users" (
	"id" varchar PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"email" text NOT NULL,
	"password" text NOT NULL,
	"role" text NOT NULL,
	"name" text NOT NULL,
	"created_at" timestamp DEFAULT now(),
	CONSTRAINT "users_email_unique" UNIQUE("email")
);
