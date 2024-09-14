import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Typography, CircularProgress, Button } from '@mui/material';

function UserProfileAnalysis() {
    const [analysis, setAnalysis] = useState('');
    const [loading, setLoading] = useState(false);

    const getProfileAnalysis = async () => {
        setLoading(true);
        try {
            const response = await axios.post('/user_profile_analysis');
            setAnalysis(response.data.analysis);
        } catch (error) {
            console.error('Error getting user profile analysis:', error);
            setAnalysis('Error analyzing user profile');
        }
        setLoading(false);
    };

    return (
        <div>
            <Typography variant="h6">User Profile Analysis</Typography>
            <Button onClick={getProfileAnalysis} disabled={loading}>
                Analyze My Profile
            </Button>
            {loading ? (
                <CircularProgress />
            ) : (
                <Typography>{analysis}</Typography>
            )}
        </div>
    );
}

export default UserProfileAnalysis;