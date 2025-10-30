interface DeepSeekMatchRequest {
  student: {
    name: string;
    university: string;
    major: string;
    graduationYear: number;
    gpa?: string;
    skills: string[];
    interests: string[];
    location: string;
    resumeText?: string;
  };
  internship: {
    title: string;
    description: string;
    requirements: string[];
    skills: string[];
    location: string;
    duration: string;
    stipend?: string;
    company: {
      name: string;
      industry: string;
      description?: string;
    };
  };
  context?: {
    currentMarketTrends?: string[];
    similarSuccessfulMatches?: string[];
    studentCareerGoals?: string;
  };
}

interface DeepSeekMatchResponse {
  overallMatch: number; // 0-100
  confidence: number; // 0-100
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

class DeepSeekAIService {
  private apiKey: string;
  private baseURL = "https://openrouter.ai/api/v1/chat/completions";
  
  constructor() {
    this.apiKey = process.env.OPENROUTER_API_KEY || '';
    if (!this.apiKey) {
      console.warn('OPENROUTER_API_KEY not found in environment variables');
    }
  }

  async analyzeMatch(request: DeepSeekMatchRequest): Promise<DeepSeekMatchResponse> {
    const prompt = this.buildMatchAnalysisPrompt(request);
    
    try {
      const response = await fetch(this.baseURL, {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${this.apiKey}`,
          "HTTP-Referer": "https://internmatch-ai.com",
          "X-Title": "InternMatch AI",
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          "model": "deepseek/deepseek-chat-v3.1:free",
          "messages": [
            {
              "role": "system",
              "content": "You are an expert career counselor and talent acquisition specialist with deep knowledge of internship matching, skill assessment, and career development. Provide detailed, actionable insights based on comprehensive analysis."
            },
            {
              "role": "user",
              "content": prompt
            }
          ],
          "temperature": 0.7,
          "max_tokens": 2000
        })
      });

      if (!response.ok) {
        throw new Error(`API request failed: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      const content = data.choices?.[0]?.message?.content;
      
      if (!content) {
        throw new Error('No content received from AI service');
      }

      return this.parseAIResponse(content);
    } catch (error) {
      console.error('DeepSeek AI analysis error:', error);
      throw new Error(`AI analysis failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  private buildMatchAnalysisPrompt(request: DeepSeekMatchRequest): string {
    return `
Analyze this student-internship match comprehensively and provide detailed insights:

STUDENT PROFILE:
- Name: ${request.student.name}
- University: ${request.student.university}
- Major: ${request.student.major}
- Graduation Year: ${request.student.graduationYear}
- GPA: ${request.student.gpa || 'Not provided'}
- Location: ${request.student.location}
- Skills: ${request.student.skills.join(', ')}
- Interests: ${request.student.interests.join(', ')}

${request.student.resumeText ? `
RESUME ANALYSIS:
${request.student.resumeText}
` : ''}

INTERNSHIP DETAILS:
- Title: ${request.internship.title}
- Company: ${request.internship.company.name} (${request.internship.company.industry})
- Location: ${request.internship.location}
- Duration: ${request.internship.duration}
- Stipend: ${request.internship.stipend || 'Not specified'}

Job Description:
${request.internship.description}

Requirements:
${request.internship.requirements.join('\n')}

Required Skills:
${request.internship.skills.join(', ')}

Company Background:
${request.internship.company.description || 'Not provided'}

${request.context ? `
CONTEXT:
- Market Trends: ${request.context.currentMarketTrends?.join(', ') || 'None provided'}
- Career Goals: ${request.context.studentCareerGoals || 'Not specified'}
` : ''}

Provide a comprehensive analysis in the following JSON format:

{
  "overallMatch": <number 0-100>,
  "confidence": <number 0-100>,
  "keyStrengths": [
    "<specific strength 1>",
    "<specific strength 2>",
    "<specific strength 3>"
  ],
  "potentialConcerns": [
    "<concern 1>",
    "<concern 2>"
  ],
  "skillGaps": [
    "<skill gap 1>",
    "<skill gap 2>"
  ],
  "careerImpact": "<detailed explanation of how this internship will impact their career>",
  "employerBenefits": [
    "<benefit to employer 1>",
    "<benefit to employer 2>",
    "<benefit to employer 3>"
  ],
  "actionableAdvice": [
    "<specific actionable advice 1>",
    "<specific actionable advice 2>",
    "<specific actionable advice 3>"
  ],
  "breakdown": {
    "skillsMatch": <number 0-100>,
    "experienceMatch": <number 0-100>,
    "locationMatch": <number 0-100>,
    "cultureMatch": <number 0-100>,
    "careerFitMatch": <number 0-100>
  }
}

Analysis Guidelines:
1. Consider both explicit skills and transferable skills from resume
2. Evaluate growth potential and learning opportunities
3. Assess cultural fit based on company industry and student background
4. Provide specific, actionable advice for improvement
5. Consider geographic and timing factors
6. Be honest about potential challenges while highlighting opportunities
7. Focus on mutual benefits for both student and employer

Return ONLY the JSON response with no additional text.
`;
  }

  private parseAIResponse(content: string): DeepSeekMatchResponse {
    try {
      // Clean the content to extract JSON
      const jsonMatch = content.match(/\{[\s\S]*\}/);
      if (!jsonMatch) {
        throw new Error('No JSON found in AI response');
      }
      
      const parsed = JSON.parse(jsonMatch[0]);
      
      // Validate required fields
      const required = ['overallMatch', 'confidence', 'keyStrengths', 'careerImpact', 'breakdown'];
      for (const field of required) {
        if (!(field in parsed)) {
          throw new Error(`Missing required field: ${field}`);
        }
      }
      
      // Ensure arrays exist
      parsed.keyStrengths = parsed.keyStrengths || [];
      parsed.potentialConcerns = parsed.potentialConcerns || [];
      parsed.skillGaps = parsed.skillGaps || [];
      parsed.employerBenefits = parsed.employerBenefits || [];
      parsed.actionableAdvice = parsed.actionableAdvice || [];
      
      // Ensure breakdown exists
      parsed.breakdown = parsed.breakdown || {
        skillsMatch: 50,
        experienceMatch: 50,
        locationMatch: 50,
        cultureMatch: 50,
        careerFitMatch: 50
      };
      
      return parsed as DeepSeekMatchResponse;
    } catch (error) {
      console.error('Failed to parse AI response:', error, 'Content:', content);
      
      // Return fallback response
      return {
        overallMatch: 50,
        confidence: 30,
        keyStrengths: ['Profile review needed'],
        potentialConcerns: ['AI analysis temporarily unavailable'],
        skillGaps: [],
        careerImpact: 'Analysis will be available once AI service is restored.',
        employerBenefits: ['Candidate shows potential'],
        actionableAdvice: ['Complete profile for better matching'],
        breakdown: {
          skillsMatch: 50,
          experienceMatch: 50,
          locationMatch: 50,
          cultureMatch: 50,
          careerFitMatch: 50
        }
      };
    }
  }

  // Analyze multiple candidates for an internship
  async analyzeCandidates(
    internship: DeepSeekMatchRequest['internship'], 
    students: DeepSeekMatchRequest['student'][]
  ): Promise<Array<DeepSeekMatchResponse & { studentId: string }>> {
    const analyses = await Promise.all(
      students.map(async (student) => {
        try {
          const analysis = await this.analyzeMatch({ student, internship });
          return { ...analysis, studentId: student.name }; // You might want to use actual ID
        } catch (error) {
          console.error(`Analysis failed for student ${student.name}:`, error);
          return {
            studentId: student.name,
            overallMatch: 0,
            confidence: 0,
            keyStrengths: [],
            potentialConcerns: ['Analysis failed'],
            skillGaps: [],
            careerImpact: 'Analysis unavailable',
            employerBenefits: [],
            actionableAdvice: [],
            breakdown: {
              skillsMatch: 0,
              experienceMatch: 0,
              locationMatch: 0,
              cultureMatch: 0,
              careerFitMatch: 0
            }
          };
        }
      })
    );

    return analyses.sort((a, b) => b.overallMatch - a.overallMatch);
  }
}

export const aiService = new DeepSeekAIService();
export type { DeepSeekMatchRequest, DeepSeekMatchResponse };
