import { useSearchParams } from "react-router-dom";

export function usePaperPanel() {
  const [params, setParams] = useSearchParams();
  const openPaperId = params.get("paper");
  const openPaper = (paperId: string) => {
    const next = new URLSearchParams(params);
    next.set("paper", paperId);
    setParams(next, { replace: false });
  };
  const closePaper = () => {
    const next = new URLSearchParams(params);
    next.delete("paper");
    setParams(next, { replace: false });
  };
  return { openPaperId, openPaper, closePaper };
}
