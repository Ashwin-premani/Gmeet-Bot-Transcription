import MatchCard from '../MatchCard';

export default function MatchCardExample() {
  return (
    <div className="p-6 space-y-6 bg-background">
      <div className="grid gap-6 md:grid-cols-2">
        {/* Internship Card */}
        <MatchCard
          type="internship"
          title="Product Manager Intern"
          company="TechFlow Solutions"
          location="San Francisco, CA"
          matchScore={87}
          matchReasons={[
            "Strong product management coursework match",
            "Previous experience with user research",
            "Skills align with job requirements",
            "Location preference matches"
          ]}
          duration="3 months"
          stipend="$3,000/month"
          skills={["Product Strategy", "User Research", "Data Analysis", "Agile", "Figma", "SQL"]}
          status="open"
          onApply={() => console.log('Apply clicked')}
        />

        {/* Candidate Card */}
        <MatchCard
          type="candidate"
          candidateName="Sarah Chen"
          university="Stanford University"
          title="Computer Science & Business"
          location="Palo Alto, CA"
          matchScore={92}
          matchReasons={[
            "Excellent academic background in CS + Business",
            "Previous PM internship experience",
            "Strong technical skills match",
            "Leadership experience in student organizations"
          ]}
          skills={["Python", "React", "Product Strategy", "Data Analysis", "Machine Learning", "Leadership"]}
          status="pending"
          onAccept={() => console.log('Accept clicked')}
          onReject={() => console.log('Reject clicked')}
        />
      </div>
    </div>
  );
}