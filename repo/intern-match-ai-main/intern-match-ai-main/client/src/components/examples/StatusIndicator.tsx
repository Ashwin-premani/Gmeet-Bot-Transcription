import StatusIndicator from '../StatusIndicator';

export default function StatusIndicatorExample() {
  return (
    <div className="p-6 space-y-6 bg-background">
      <div className="grid gap-4 md:grid-cols-3">
        <div className="space-y-4">
          <h3 className="font-medium">Application Status</h3>
          <div className="space-y-3">
            <StatusIndicator status="pending" />
            <StatusIndicator status="accepted" />
            <StatusIndicator status="rejected" />
          </div>
        </div>

        <div className="space-y-4">
          <h3 className="font-medium">Internship Status</h3>
          <div className="space-y-3">
            <StatusIndicator status="open" />
            <StatusIndicator status="closed" />
            <StatusIndicator status="filled" />
          </div>
        </div>

        <div className="space-y-4">
          <h3 className="font-medium">With Progress</h3>
          <div className="space-y-3">
            <StatusIndicator status="pending" showProgress progress={25} />
            <StatusIndicator status="open" showProgress progress={67} />
            <StatusIndicator status="filled" showProgress progress={100} />
          </div>
        </div>
      </div>
    </div>
  );
}