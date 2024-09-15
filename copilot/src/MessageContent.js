import React from 'react';
import ReactMarkdown from 'react-markdown';
import { Typography, Paper, Box } from '@mui/material';

const JSONDisplay = ({ data }) => {
    const renderJSONValue = (value) => {
        if (typeof value === 'object' && value !== null) {
            return <JSONDisplay data={value} />;
        }
        return <span>{JSON.stringify(value)}</span>;
    };

    return (
        <Box component="pre" sx={{ m: 0, p: 1, bgcolor: 'background.paper', borderRadius: 1, overflow: 'auto' }}>
            {'{'}
            {Object.entries(data).map(([key, value], index, array) => (
                <Box key={key} sx={{ pl: 2 }}>
                    <Typography component="span" color="primary">"{key}"</Typography>: {renderJSONValue(value)}
                    {index < array.length - 1 ? ',' : ''}
                </Box>
            ))}
            {'}'}
        </Box>
    );
};

const MessageContent = ({ content }) => {
    const isJSON = (str) => {
        try {
            JSON.parse(str);
            return true;
        } catch (e) {
            return false;
        }
    };

    if (isJSON(content)) {
        const jsonData = JSON.parse(content);
        return <JSONDisplay data={jsonData} />;
    }

    // Pour le contenu non-JSON, on utilise ReactMarkdown comme avant
    return (
        <Paper sx={{ p: 2, mt: 1, maxWidth: '100%', overflowX: 'auto' }}>
            <ReactMarkdown
                components={{
                    h1: ({ node, ...props }) => <Typography variant="h4" gutterBottom {...props} />,
                    h2: ({ node, ...props }) => <Typography variant="h5" gutterBottom {...props} />,
                    h3: ({ node, ...props }) => <Typography variant="h6" gutterBottom {...props} />,
                    p: ({ node, ...props }) => <Typography paragraph {...props} />,
                    li: ({ node, ...props }) => <Typography component="li" sx={{ ml: 2 }} {...props} />,
                    ul: ({ node, ...props }) => <Box component="ul" sx={{ pl: 2 }} {...props} />,
                    code: ({ node, inline, ...props }) =>
                        inline ? (
                            <Box component="code" sx={{ bgcolor: 'grey.100', p: 0.5, borderRadius: 1 }} {...props} />
                        ) : (
                            <Box component="pre" sx={{ p: 1, bgcolor: 'grey.100', borderRadius: 1, overflow: 'auto' }}>
                                <code {...props} />
                            </Box>
                        ),
                }}
            >
                {content}
            </ReactMarkdown>
        </Paper>
    );
};

export default MessageContent;