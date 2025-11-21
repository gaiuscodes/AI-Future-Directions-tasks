// app/api/vehicles/route.ts
// Backend: Vehicle API endpoints

import { NextRequest, NextResponse } from 'next/server';
import { getServerSession } from '@/lib/auth';
import { getVehicles, createVehicle } from '@/services/vehicle.service';
import { createErrorResponse, createSuccessResponse } from '@/utils/errors';
import { z } from 'zod';

// GET /api/vehicles - List vehicles with search and filters (PUBLIC - no auth required)
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    
    const filters = {
      route: searchParams.get('route') || undefined,
      sacco: searchParams.get('sacco') || undefined,
      plate: searchParams.get('plate') || undefined,
      minRating: searchParams.get('minRating') || undefined,
      limit: searchParams.get('limit') ? parseInt(searchParams.get('limit')!) : undefined,
      offset: searchParams.get('offset') ? parseInt(searchParams.get('offset')!) : undefined,
    };

    const result = await getVehicles(filters);

    return NextResponse.json(
      createSuccessResponse(result)
    );

  } catch (error) {
    console.error('Fetch vehicles error:', error);
    return NextResponse.json(
      createErrorResponse(
        'INTERNAL_ERROR',
        'An unexpected error occurred',
        500
      ),
      { status: 500 }
    );
  }
}

// POST /api/vehicles - Create vehicle (Admin only)
export async function POST(request: NextRequest) {
  try {
    const session = await getServerSession();
    
    if (!session?.user || session.user.role !== 'ADMIN') {
      return NextResponse.json(
        createErrorResponse(
          'FORBIDDEN',
          'Admin access required',
          403
        ),
        { status: 403 }
      );
    }

    const body = await request.json();
    const vehicle = await createVehicle(body);

    return NextResponse.json(
      createSuccessResponse(
        vehicle,
        'Vehicle created successfully'
      ),
      { status: 201 }
    );

  } catch (error) {
    console.error('Create vehicle error:', error);

    if (error instanceof z.ZodError) {
      return NextResponse.json(
        createErrorResponse(
          'VALIDATION_ERROR',
          'Invalid input data',
          400,
          error.errors.map(e => ({
            field: e.path.join('.'),
            message: e.message
          }))
        ),
        { status: 400 }
      );
    }

    if (error instanceof Error && error.message === 'VEHICLE_EXISTS') {
      return NextResponse.json(
        createErrorResponse(
          'VEHICLE_EXISTS',
          'Vehicle with this registration plate already exists',
          409
        ),
        { status: 409 }
      );
    }

    return NextResponse.json(
      createErrorResponse(
        'INTERNAL_ERROR',
        'An unexpected error occurred',
        500
      ),
      { status: 500 }
    );
  }
}