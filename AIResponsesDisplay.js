import React from 'react';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion"

const AIResponsesDisplay = ({ responses }) => {
  return (
    <div className="p-4 bg-white rounded-lg shadow">
      <h2 className="text-2xl font-bold mb-4">AI Model Responses</h2>
      <Accordion type="single" collapsible>
        {Object.entries(responses).map(([model, response], index) => (
          <AccordionItem value={`item-${index}`} key={index}>
            <AccordionTrigger className="text-lg font-semibold">
              {model}
            </AccordionTrigger>
            <AccordionContent>
              <p className="mt-2">{response}</p>
            </AccordionContent>
          </AccordionItem>
        ))}
      </Accordion>
    </div>
  );
};

export default AIResponsesDisplay;